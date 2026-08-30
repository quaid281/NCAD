"""Conditional Flow Matching Time-Series Joint Embedding Predictive Architecture (FlowTSJEPA).

Replaces deterministic latent regression with continuous Optimal Transport Flow Matching (OT-CFM).
Instead of collapsing multi-modal future trajectories to a conditional mean, FlowTSJEPA learns
a continuous neural velocity field v_psi(z_t, t, z_ctx) that transports Gaussian prior noise z_0 ~ N(0, I)
along straight ODE paths to the ground-truth future target latent representation z_1 = E_phi(x_tgt).
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
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
) -> Tuple[torch.Tensor, dict]:
    """Combined Optimal Transport Flow Matching and Non-Contrastive VICReg Loss.
    
    1. Flow Matching Objective: || v_pred - (z_1 - z_0) ||^2
    2. Variance Regularization: Prevents context and target representations from collapsing.
    3. Covariance Decorrelation: Prevents dimensional collapse.
    """
    # 1. Flow Matching Loss
    loss_flow = F.mse_loss(v_pred, v_target)

    # 2. Branch-wise VICReg Regularization
    var_loss = torch.tensor(0.0, device=v_pred.device)
    cov_loss = torch.tensor(0.0, device=v_pred.device)

    branch_count = 0
    if z_ctx is not None:
        v_c, c_c = _vicreg_branch_loss(z_ctx, gamma=gamma, eps=eps)
        var_loss = var_loss + v_c
        cov_loss = cov_loss + c_c
        branch_count += 1

    if z_tgt_true is not None:
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


class FlowTSJEPAModel(nn.Module):
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
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        self.flow_predictor = FlowLatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_layers,
            dropout=dropout,
        )

        # Buffers for Mahalanobis-whitened flow scoring
        self.register_buffer("precision_matrix", torch.eye(latent_dim))
        self.register_buffer("residual_mean", torch.zeros(latent_dim))
        self.register_buffer("precision_fitted", torch.tensor(False, dtype=torch.bool))

    def train(self, mode: bool = True) -> FlowTSJEPAModel:
        """Ensure target encoder strictly remains in eval mode during training."""
        super().train(mode)
        self.target_encoder.eval()
        return self

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
            t = torch.rand(B, device=device)

        # Sample prior noise z_0 ~ N(0, I) if not provided
        if z_noise is None:
            z_noise = torch.randn(B, self.latent_dim, device=device)

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
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Update target encoder weights and buffers via Exponential Moving Average (EMA)."""
        m = self.ema_decay if decay is None else decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)
        for buf_q, buf_k in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            buf_k.data.copy_(buf_q.data)

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
        z_ctx = self.context_encoder(context_windows)

        if z_init is None:
            z_t = torch.randn(B, self.latent_dim, device=device)
        else:
            z_t = z_init.clone()

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_curr = torch.full((B,), t_val, device=device, dtype=torch.float32)

            if solver == "euler":
                v = self.flow_predictor(z_t, t_curr, z_ctx)
                z_t = z_t + dt * v

            elif solver == "midpoint":
                # 2nd order Runge-Kutta / Midpoint method
                v_curr = self.flow_predictor(z_t, t_curr, z_ctx)
                z_mid = z_t + 0.5 * dt * v_curr
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=torch.float32)
                v_mid = self.flow_predictor(z_mid, t_mid, z_ctx)
                z_t = z_t + dt * v_mid

            elif solver == "rk4":
                # 4th order classic Runge-Kutta
                k1 = self.flow_predictor(z_t, t_curr, z_ctx)
                t_half = torch.full((B,), t_val + 0.5 * dt, device=device, dtype=torch.float32)
                k2 = self.flow_predictor(z_t + 0.5 * dt * k1, t_half, z_ctx)
                k3 = self.flow_predictor(z_t + 0.5 * dt * k2, t_half, z_ctx)
                t_next = torch.full((B,), t_val + dt, device=device, dtype=torch.float32)
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
    ) -> torch.Tensor:
        """Compute deterministic Optimal Transport midpoint flow discrepancy.
        
        Evaluates the vector field consistency at t=0.5 along the straight trajectory
        from prior mean z_0=0 to observed target z_1=E_phi(x_tgt):
        || v_psi(0.5 * z_tgt, t=0.5, z_ctx) - z_tgt ||.
        
        Completely eliminates Monte Carlo sampling noise, enabling precise GPD tail calibration.
        
        Args:
            context_windows: (B, L_ctx, C)
            observed_target_windows: (B, L_tgt, C)
            use_mahalanobis: Whether to apply covariance whitening
            
        Returns:
            Discrepancy scores of shape (B,)
        """
        self.eval()
        B = context_windows.size(0)
        device = context_windows.device

        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        t_mid = torch.full((B,), 0.5, device=device, dtype=torch.float32)
        z_mid = 0.5 * z_tgt
        v_pred = self.flow_predictor(z_mid, t_mid, z_ctx)
        diff = v_pred - z_tgt

        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_c = diff - self.residual_mean
            m_dist = torch.sum((diff_c @ self.precision_matrix) * diff_c, dim=-1)
            return torch.sqrt(torch.clamp(m_dist, min=1e-8))
        else:
            return torch.linalg.norm(diff, dim=-1)

    @torch.no_grad()
    def compute_instantaneous_energy_discrepancy(
        self,
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        n_eval_times: int = 3,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        """Compute deterministic predictive discrepancy (alias for backward compatibility)."""
        return self.compute_predictive_discrepancy(context_windows, target_windows, use_mahalanobis=use_mahalanobis)

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
        context_windows: torch.Tensor,
        target_windows: torch.Tensor,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical vector field residual covariance for Mahalanobis whitened scoring."""
        self.eval()
        n_samples = len(context_windows)
        residuals_list = []

        for i in range(0, n_samples, batch_size):
            ctx_b = context_windows[i : i + batch_size]
            tgt_b = target_windows[i : i + batch_size]
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            B_b = len(ctx_b)
            t_mid = torch.full((B_b,), 0.5, device=ctx_b.device, dtype=torch.float32)
            z_mid = 0.5 * z_tgt
            v_pred = self.flow_predictor(z_mid, t_mid, z_ctx)
            residuals_list.append(v_pred - z_tgt)

        residuals = torch.cat(residuals_list, dim=0)
        mean_res = residuals.mean(dim=0, keepdim=True)
        self.residual_mean.copy_(mean_res.squeeze(0))
        residuals_centered = residuals - mean_res
        cov = (residuals_centered.T @ residuals_centered) / max(len(residuals) - 1, 1)
        cov_reg = cov + reg * torch.eye(self.latent_dim, device=cov.device)
        self.precision_matrix.copy_(torch.linalg.pinv(cov_reg))
        self.precision_fitted.copy_(torch.tensor(True, dtype=torch.bool))


# Alias
FlowTSJEPA = FlowTSJEPAModel
