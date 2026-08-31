"""Conditional Flow Matching Time-Series Joint Embedding Predictive Architecture (FlowTSJEPA).

Replaces deterministic latent regression with continuous Optimal Transport Flow Matching (OT-CFM).
Instead of collapsing multi-modal future trajectories to a conditional mean, FlowTSJEPA learns
a continuous neural velocity field v_psi(z_t, t, z_ctx) that transports Gaussian prior noise z_0 ~ N(0, I)
along straight ODE paths to the ground-truth future target latent representation z_1 = E_phi(x_tgt).
"""

from __future__ import annotations

import math
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase
from src.models.ts_jepa import _vicreg_branch_loss


class TimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous flow time t in [0, 1]."""

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for time t.
        
        Args:
            t: Tensor of shape (B,) with continuous time values in [0, 1].
            
        Returns:
            Tensor of shape (B, embed_dim).
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)
        dtype = self.mlp[0].weight.dtype
        device = t.device
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=dtype, device=device) / half_dim
        )
        args = t[:, None].to(dtype=dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class FlowLatentPredictor(nn.Module):
    """Continuous Velocity Field Predictor v_psi(z_t, t, z_ctx).
    
    Predicts the instantaneous velocity dz_t / dt in the target latent representation space,
    conditioned on time t in [0, 1] and historical context representation z_ctx.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimestepEmbedding(embed_dim=hidden_dim)

        # Joint input projection: [z_t, z_ctx] -> hidden_dim
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        # Residual MLP blocks with time conditioning modulation
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "linear1": nn.Linear(hidden_dim, hidden_dim),
                        "act": nn.GELU(),
                        "linear2": nn.Linear(hidden_dim, hidden_dim),
                        "dropout": nn.Dropout(dropout),
                        "time_proj": nn.Linear(hidden_dim, hidden_dim),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity: (B, D) x (B,) x (B, D) -> (B, D).
        
        Args:
            z_t: Intermediate state in latent space at time t, shape (B, D)
            t: Continuous time in [0, 1], shape (B,) or scalar
            z_ctx: Conditioning context representation, shape (B, D)
            
        Returns:
            v_t: Velocity vector dz_t/dt, shape (B, D)
        """
        if t.ndim == 0 or (t.ndim == 1 and t.size(0) == 1 and z_t.size(0) > 1):
            t = t.expand(z_t.size(0))

        t_emb = self.time_embed(t)  # (B, hidden_dim)
        x = self.input_proj(torch.cat([z_t, z_ctx], dim=-1))  # (B, hidden_dim)

        for block in self.blocks:
            residual = x
            h = block["norm"](x)
            # Add time conditioning modulation
            h = h + block["time_proj"](t_emb)
            h = block["act"](block["linear1"](h))
            h = block["dropout"](block["linear2"](h))
            x = residual + h

        x = self.out_norm(x)
        return self.out_proj(x)


def von_neumann_operator_entropy_loss(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Von Neumann / Operator Bregman Divergence loss for covariance decorrelation.

    Computes D_Breg(rho || (1/D) * I) = Tr(rho log rho) + log(D) where rho is the normalized
    batch covariance matrix (density operator).

    Rooted in Quantum Information Theory and Operator Convexity (Petz, 2007):
    forces all covariance eigenvalues to be identical (maximal entropy / isotropic dispersion),
    with logarithmic gradient steepness near zero singular values preventing dimensional collapse.

    Args:
        z: Latent embeddings of shape (B, D) or (B, N, D).
        eps: Small eigenvalue regularizer for log stability.

    Returns:
        Scalar loss >= 0, achieving 0 iff all covariance eigenvalues are equal.
    """
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    D = z.size(-1)
    N = z.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=z.device)

    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / (N - 1)
    tr = torch.trace(cov) + eps
    rho = cov / tr + (eps / D) * torch.eye(D, device=z.device, dtype=z.dtype)
    rho = rho / torch.trace(rho)
    evals = torch.linalg.eigvalsh(rho)
    evals = torch.clamp(evals, min=eps)
    p = evals / evals.sum()
    vn_entropy = -torch.sum(p * torch.log(p))
    max_entropy = math.log(float(D))
    return torch.clamp(max_entropy - vn_entropy, min=0.0)


def flow_matching_vicreg_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    z_ctx: Optional[torch.Tensor] = None,
    z_tgt_true: Optional[torch.Tensor] = None,
    flow_weight: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 0.1,
    gamma: float = 1.0,
    eps: float = 1e-4,
    reg_mode: Literal["vicreg", "operator_entropy"] = "operator_entropy",
) -> Tuple[torch.Tensor, dict]:
    """Combined Optimal Transport Flow Matching and Manifold Regularization Loss.
    
    1. Flow Matching Objective: || v_pred - (z_1 - z_0) ||^2
    2. Variance Regularization: Prevents context and target representations from collapsing.
    3. Covariance Regularization:
       - 'operator_entropy' (default): Quantum Von Neumann Operator Entropy divergence.
       - 'vicreg': Classical off-diagonal covariance Frobenius norm penalty.
    """
    # 1. Flow Matching Loss
    loss_flow = F.mse_loss(v_pred, v_target)

    # 2. Branch-wise Variance and Covariance Regularization
    var_loss = torch.tensor(0.0, device=v_pred.device)
    cov_loss = torch.tensor(0.0, device=v_pred.device)

    branch_count = 0
    if z_ctx is not None:
        if reg_mode == "operator_entropy":
            z_c_flat = z_ctx.reshape(-1, z_ctx.size(-1))
            std_c = torch.sqrt(torch.var(z_c_flat, dim=0, unbiased=False) + eps)
            v_c = torch.mean(F.relu(gamma - std_c))
            c_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)
        else:
            v_c, c_c = _vicreg_branch_loss(z_ctx, gamma=gamma, eps=eps)
        var_loss = var_loss + v_c
        cov_loss = cov_loss + c_c
        branch_count += 1

    if z_tgt_true is not None:
        if reg_mode == "operator_entropy":
            z_t_flat = z_tgt_true.reshape(-1, z_tgt_true.size(-1))
            std_t = torch.sqrt(torch.var(z_t_flat, dim=0, unbiased=False) + eps)
            v_t = torch.mean(F.relu(gamma - std_t))
            c_t = von_neumann_operator_entropy_loss(z_tgt_true, eps=eps)
        else:
            v_t, c_t = _vicreg_branch_loss(z_tgt_true, gamma=gamma, eps=eps)
        var_loss = var_loss + v_t
        cov_loss = cov_loss + c_t
        branch_count += 1

    if branch_count > 0:
        var_loss = var_loss / branch_count
        cov_loss = cov_loss / branch_count

    total_loss = flow_weight * loss_flow + var_weight * var_loss + cov_weight * cov_loss
    loss_metrics = {
        "total_loss": float(total_loss.item()),
        "loss_flow": float(loss_flow.item()),
        "loss_var": float(var_loss.item()),
        "loss_cov": float(cov_loss.item()),
    }
    return total_loss, loss_metrics


def _get_chebyshev_collocation_nodes(
    mode: Literal["midpoint", "chebyshev_3", "chebyshev_5"] = "midpoint",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[float], List[float]]:
    """Compute deterministic quadrature collocation nodes and weights along the OT path t in (0, 1)."""
    if mode == "midpoint":
        return [0.5], [1.0]
    elif mode == "chebyshev_3":
        t1 = 0.5 * (1.0 - math.sqrt(2) / 2.0)
        t2 = 0.5
        t3 = 0.5 * (1.0 + math.sqrt(2) / 2.0)
        w = [0.2761, 0.4478, 0.2761]
        sum_w = sum(w)
        return [t1, t2, t3], [x / sum_w for x in w]
    elif mode == "chebyshev_5":
        nodes = [0.5 * (1.0 + math.cos((5 - k) * math.pi / 5.0)) for k in range(1, 5)]
        w = [0.2, 0.3, 0.3, 0.2]
        sum_w = sum(w)
        return nodes, [x / sum_w for x in w]
    else:
        raise ValueError(f"Unknown collocation mode: {mode}. Choose 'midpoint', 'chebyshev_3', or 'chebyshev_5'.")


class FlowTSJEPAModel(JEPABase):
    """Conditional Flow Matching Time-Series Joint Embedding Predictive Architecture.
    
    Wraps:
    - Context Encoder E_theta (active gradients)
    - Target Encoder E_phi (EMA updated, strictly deterministic eval mode)
    - Flow Latent Predictor v_psi (trained via Optimal Transport Flow Matching)
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int,
        predictor_hidden_dim: int = 64,
        predictor_layers: int = 3,
        ema_decay: float = 0.995,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Target encoder is an EMA copy of context encoder
        self.target_encoder = self.init_target_encoder(context_encoder)

        self.flow_predictor = FlowLatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # Buffers for Mahalanobis-whitened flow scoring
        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for flow matching training.
        
        Args:
            context_windows: Tensor of shape (B, L_ctx, C)
            target_windows: Optional tensor of shape (B, L_tgt, C)
            t: Optional continuous time in [0, 1], shape (B,). If None, sampled uniformly U(0, 1).
            z_noise: Optional Gaussian prior noise at t=0, shape (B, D). If None, sampled from N(0, I).
            
        Returns:
            Tuple of (z_ctx, z_tgt_true, v_pred, v_target)
        """
        B = context_windows.size(0)
        device = context_windows.device

        # Encode context representation
        z_ctx = self.context_encoder(context_windows)

        if target_windows is None:
            return z_ctx, None, None, None

        # Encode ground truth target representation with EMA target encoder
        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)  # z_1 at t=1

        # Sample time t in [0, 1] if not provided
        if t is None:
            t = torch.rand(B, device=device, dtype=context_windows.dtype)

        # Sample prior noise z_0 ~ N(0, I) if not provided
        if z_noise is None:
            z_noise = torch.randn(B, self.latent_dim, device=device, dtype=context_windows.dtype)

        # Optimal Transport Flow Matching Linear Interpolation:
        # z_t = (1 - t) * z_0 + t * z_1
        t_expand = t.view(B, 1)
        z_t = (1.0 - t_expand) * z_noise + t_expand * z_tgt_true

        # Ground truth target velocity: u_t = dz_t / dt = z_1 - z_0
        v_target = z_tgt_true - z_noise

        # Predicted velocity from the continuous vector field
        v_pred = self.flow_predictor(z_t, t, z_ctx)

        return z_ctx, z_tgt_true, v_pred, v_target

    @torch.no_grad()
    def sample_target(
        self,
        context_windows: torch.Tensor,
        n_steps: int = 4,
        solver: Literal["euler", "midpoint", "rk4"] = "midpoint",
        z_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate future target latent embeddings by integrating the ODE from t=0 to t=1.
        
        Args:
            context_windows: (B, L_ctx, C)
            n_steps: Number of integration steps (1-4 is typically sufficient for OT flows)
            solver: Numerical ODE solver ('euler', 'midpoint', 'rk4')
            z_init: Optional initial noise at t=0, shape (B, D). Defaults to N(0, I).
            
        Returns:
            z_sampled: Generated future target latent representations (B, D)
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype
        z_ctx = self.context_encoder(context_windows)

        if z_init is None:
            z_t = torch.randn(B, self.latent_dim, device=device, dtype=dtype)
        else:
            z_t = z_init.clone()

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_curr = torch.full((B,), t_val, device=device, dtype=dtype)

            if solver == "euler":
                v = self.flow_predictor(z_t, t_curr, z_ctx)
                z_t = z_t + dt * v

            elif solver == "midpoint":
                # 2nd order Runge-Kutta / Midpoint method
                v_curr = self.flow_predictor(z_t, t_curr, z_ctx)
                z_mid = z_t + 0.5 * dt * v_curr
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=dtype)
                v_mid = self.flow_predictor(z_mid, t_mid, z_ctx)
                z_t = z_t + dt * v_mid

            elif solver == "rk4":
                # 4th order classic Runge-Kutta
                k1 = self.flow_predictor(z_t, t_curr, z_ctx)
                t_half = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=dtype)
                k2 = self.flow_predictor(z_t + 0.5 * dt * k1, t_half, z_ctx)
                k3 = self.flow_predictor(z_t + 0.5 * dt * k2, t_half, z_ctx)
                t_next = torch.full((B,), t_val + dt, device=device, dtype=dtype)
                k4 = self.flow_predictor(z_t + dt * k3, t_next, z_ctx)
                z_t = z_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(f"Unknown solver: {solver}. Choose 'euler', 'midpoint', or 'rk4'.")

        return z_t

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
        collocation: Literal["midpoint", "chebyshev_3", "chebyshev_5"] = "midpoint",
    ) -> torch.Tensor:
        """Compute deterministic Optimal Transport flow discrepancy.
        
        Evaluates the continuous velocity field consistency along the straight OT-CFM trajectory
        interpolating from the prior mean (z_0 = 0) to observed target z_1 = E_phi(x_tgt):
        z_t = t * z_tgt, with target velocity v_target = z_tgt - z_0 = z_tgt.
        
        Evaluates at deterministic Chebyshev-Lobatto quadrature nodes t_k in (0, 1) (default: t=0.5 midpoint).
        This eliminates Monte Carlo sampling noise while capturing non-linear vector-field curvature,
        enabling exact Extreme Value Theory (EVT) Generalized Pareto tail calibration.
        
        Args:
            context_windows: (B, L_ctx, C)
            observed_target_windows: (B, L_tgt, C)
            use_mahalanobis: Whether to apply covariance whitening
            collocation: 'midpoint' (1-point t=0.5), 'chebyshev_3' (3-point), or 'chebyshev_5' (5-point)
            
        Returns:
            Discrepancy scores of shape (B,)
        """
        if use_mahalanobis and not bool(self.precision_fitted.item()):
            raise RuntimeError("Mahalanobis scoring requested, but covariance has not been fitted.")

        self.eval()
        B = context_windows.size(0)
        device = context_windows.device
        dtype = context_windows.dtype

        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        nodes, weights = _get_chebyshev_collocation_nodes(collocation, device=device, dtype=dtype)
        total_score = torch.zeros(B, device=device, dtype=dtype)

        for t_val, w_val in zip(nodes, weights):
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            z_node = t_val * z_tgt
            v_pred = self.flow_predictor(z_node, t_tensor, z_ctx)
            diff = v_pred - z_tgt

            if use_mahalanobis:
                diff_c = diff - self.residual_mean
                m_dist = torch.sum((diff_c @ self.precision_matrix) * diff_c, dim=-1)
                node_score = torch.sqrt(torch.clamp(m_dist, min=1e-8))
            else:
                node_score = torch.linalg.norm(diff, dim=-1)

            total_score = total_score + w_val * node_score

        return total_score

    @torch.no_grad()
    def compute_instantaneous_energy_discrepancy(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        n_eval_times: int = 3,
        use_mahalanobis: bool = False,
        collocation: Literal["midpoint", "chebyshev_3", "chebyshev_5"] = "midpoint",
    ) -> torch.Tensor:
        """Compute deterministic predictive discrepancy (alias with collocation support)."""
        return self.compute_predictive_discrepancy(
            context_windows, target_windows, use_mahalanobis=use_mahalanobis, collocation=collocation
        )


    @torch.no_grad()
    def compute_trajectory_discrepancy(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        n_steps: int = 4,
        solver: Literal["euler", "midpoint", "rk4"] = "midpoint",
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Compute the trajectory discrepancy || E_phi(x_tgt) - ODE_Sample(z_ctx) ||.
        
        Args:
            context_windows: (B, L_ctx, C)
            target_windows: (B, L_tgt, C)
            n_steps: Number of integration steps (default: 4)
            solver: Numerical ODE solver
            n_samples: Number of stochastic trajectories to sample and average
            
        Returns:
            Discrepancy scores of shape (B,)
        """
        self.eval()
        z_tgt_obs = self.target_encoder(target_windows)

        scores = []
        for _ in range(n_samples):
            z_gen = self.sample_target(context_windows, n_steps=n_steps, solver=solver)
            dist = torch.linalg.norm(z_tgt_obs - z_gen, dim=-1)
            scores.append(dist)

        return torch.stack(scores, dim=0).mean(dim=0)

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical vector field residual covariance for Mahalanobis whitened scoring.

        Accepts numpy arrays or tensors; data is transferred in batches.
        """
        from src.models._jepa_utils import fit_covariance_batched

        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = ctx_b.size(0)
            t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=ctx_b.dtype)
            z_mid = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_mid, t_mid, z_ctx)
            return v_pred - z_tgt

        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.latent_dim,
            batch_size=batch_size,
            reg=reg,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )


# Alias
FlowTSJEPA = FlowTSJEPAModel
