"""Harmonic Spring Joint Embedding Predictive Architecture (HarmonicSpringJEPA).

The minimalist, closed-form successor to Potential-Flow JEPA:
Eliminates continuous-time generative scaffolding (t in [0, 1], prior noise z_0,
autograd in forward passes, ODE integrators, and Hutchinson approximations).

Models nominal dynamics as a state-dependent Harmonic Potential Well:
    Phi(z_tgt | z_ctx) = 1/2 * (z_tgt - mu_ctx)^T M(z_ctx) (z_tgt - mu_ctx)

Where:
- mu_ctx = f_theta(z_ctx) in R^D is the predicted attractor center.
- M(z_ctx) = L L^T + diag(d) + eps * I_D > 0 is the Riemannian stiffness metric tensor.
- Energy Curvature (Laplacian) is exact and closed-form in O(1):
    Delta Phi = Tr(M(z_ctx)) = ||L||_F^2 + sum(d) + D * eps
- Restoring Velocity is exact and closed-form in O(D * rank):
    v = -nabla_z Phi = -L (L^T e) - d * e - eps * e
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss
from src.models.jepa.ts_jepa import _vicreg_branch_loss


class HarmonicSpringHead(nn.Module):
    """Predicts attractor center mu(z_ctx) and Riemannian stiffness matrix M(z_ctx).
    
    Parameterizes M(z_ctx) via:
    - Low-rank Cholesky factor L in R^{D x rank}
    - Positive diagonal vector d in R^D_{> 0}
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        rank: int = 4,
        eps: float = 1e-3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.eps = eps

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Attractor mean prediction: mu in R^D
        self.mu_head = nn.Linear(hidden_dim, latent_dim)

        # Low-rank factor: L in R^{D x rank}
        self.L_head = nn.Linear(hidden_dim, latent_dim * rank)

        # Diagonal stiffness: d in R^D (softplus activated)
        self.d_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict attractor mean, low-rank stiffness factor, and diagonal stiffness.
        
        Args:
            z_ctx: Context representation, shape (B, D)
            
        Returns:
            mu: Attractor center, shape (B, D)
            L: Low-rank Cholesky stiffness factor, shape (B, D, rank)
            d: Diagonal mode stiffness, shape (B, D)
        """
        B = z_ctx.size(0)
        h = self.mlp(z_ctx)

        mu = self.mu_head(h)
        L = self.L_head(h).view(B, self.latent_dim, self.rank)
        # Softplus ensures positive diagonal entries d_j > 0
        d = F.softplus(self.d_head(h)) + self.eps

        return mu, L, d


class HarmonicSpringJEPAModel(JEPABase):
    """Harmonic Spring Joint Embedding Predictive Architecture (Spring-JEPA).
    
    A closed-form, zero-sampling-latency model combining:
    1. Causal Context Encoder E_theta
    2. Target Momentum Encoder E_phi (EMA replica)
    3. Harmonic Spring Head predicting attractor center mu and stiffness tensor M
    4. Closed-form Energy and Curvature evaluation without ODE integration or autograd.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int,
        hidden_dim: int = 64,
        rank: int = 4,
        eps: float = 1e-3,
        ema_decay: float = 0.995,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.rank = rank
        self.eps = eps
        self.ema_decay = ema_decay

        # Target encoder is an EMA copy of context encoder
        self.target_encoder = self.init_target_encoder(context_encoder)

        # Harmonic Spring Head
        self.spring_head = HarmonicSpringHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            eps=eps,
            dropout=dropout,
        )

        # Running statistics for curvature z-score normalization
        self.register_buffer("curv_mean", torch.tensor(0.0))
        self.register_buffer("curv_var", torch.tensor(1.0))
        self.register_buffer("curv_count", torch.tensor(0.0))

        # Buffers for Mahalanobis-whitened residual scoring
        self.register_mahalanobis_buffers(latent_dim)

    def compute_energy_and_curvature(
        self,
        z_ctx: torch.Tensor,
        z_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute exact potential energy, closed-form Laplacian curvature, and log-determinant.
        
        Args:
            z_ctx: Context latent vector, shape (B, D)
            z_tgt: Target latent vector, shape (B, D)
            
        Returns:
            energy: Potential energy e^T M e, shape (B,)
            curvature: Exact Laplacian Tr(M), shape (B,)
            log_det: Exact log-determinant log det(M), shape (B,)
        """
        B, D = z_ctx.shape
        mu, L, d = self.spring_head(z_ctx)  # (B, D), (B, D, r), (B, D)

        # Residual deviation: e = z_tgt - mu
        e = z_tgt - mu  # (B, D)

        # 1. Potential Energy: e^T (L L^T + diag(d)) e = ||L^T e||_2^2 + sum(d_j * e_j^2)
        # L^T @ e -> shape (B, r)
        Lt_e = torch.bmm(L.transpose(1, 2), e.unsqueeze(-1)).squeeze(-1)  # (B, r)
        quad_low_rank = torch.sum(Lt_e ** 2, dim=-1)  # (B,)
        quad_diag = torch.sum(d * (e ** 2), dim=-1)  # (B,)
        energy = quad_low_rank + quad_diag  # (B,)

        # 2. Exact Closed-Form Curvature (Laplacian): Tr(M) = ||L||_F^2 + sum(d)
        curv_low_rank = torch.sum(L ** 2, dim=[1, 2])  # (B,)
        curv_diag = torch.sum(d, dim=-1)  # (B,)
        curvature = curv_low_rank + curv_diag  # (B,)

        # 3. Exact Log-Determinant via Matrix Determinant Lemma:
        # det(diag(d) + L L^T) = det(I_r + L^T diag(d)^{-1} L) * prod(d_j)
        # log det(M) = sum(log d_j) + log det(I_r + L^T diag(d)^{-1} L)
        log_det_diag = torch.sum(torch.log(d), dim=-1)  # (B,)
        # Scale L by 1/sqrt(d): (B, D, r) * (B, D, 1)
        L_scaled = L / torch.sqrt(d.unsqueeze(-1))  # (B, D, r)
        # Capacitance matrix: C = I_r + L_scaled^T @ L_scaled in R^{r x r}
        C = torch.eye(self.rank, device=z_ctx.device, dtype=z_ctx.dtype).unsqueeze(0) + torch.bmm(
            L_scaled.transpose(1, 2), L_scaled
        )  # (B, r, r)
        evals_c = torch.linalg.eigvalsh(C)
        log_det_cap = torch.sum(torch.log(torch.clamp(evals_c, min=1e-6)), dim=-1)
        log_det = log_det_diag + log_det_cap  # (B,)

        return energy, curvature, log_det

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for harmonic spring predictive learning."""
        z_ctx = self.context_encoder(context_windows)

        if target_windows is None:
            return z_ctx, None, None, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.target_encoder(target_windows)

        energy, curvature, log_det = self.compute_energy_and_curvature(z_ctx, z_tgt_true)

        return z_ctx, z_tgt_true, energy, curvature, log_det

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        det_weight: float = 0.05,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Harmonic Spring potential objective with log-det stiffness regularization."""
        z_ctx, z_tgt_true, energy, curvature, log_det = self.forward(ctx, tgt)

        # 1. Harmonic Energy Minimization: E[Phi(z_tgt | z_ctx)]
        loss_energy = torch.mean(energy)

        # 2. Anti-Stiffness Collapse Barrier: -1/D * E[log det(M)]
        loss_det = -torch.mean(log_det) / float(self.latent_dim)

        # 3. Representation Non-Collapse (VICReg / Operator Entropy)
        z_c_flat = z_ctx.reshape(-1, z_ctx.size(-1))
        std_c = torch.sqrt(torch.var(z_c_flat, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        z_t_flat = z_tgt_true.reshape(-1, z_tgt_true.size(-1))
        std_t = torch.sqrt(torch.var(z_t_flat, dim=0, unbiased=False) + eps)
        var_t = torch.mean(F.relu(gamma - std_t))
        cov_t = von_neumann_operator_entropy_loss(z_tgt_true, eps=eps)

        loss_var = 0.5 * (var_c + var_t)
        loss_cov = 0.5 * (cov_c + cov_t)

        total_loss = loss_energy + det_weight * loss_det + var_weight * loss_var + cov_weight * loss_cov

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_energy": float(loss_energy.item()),
            "loss_det": float(loss_det.item()),
            "loss_var": float(loss_var.item()),
            "loss_cov": float(loss_cov.item()),
            "mean_curvature": float(torch.mean(curvature).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
        include_curvature: bool = True,
        curvature_weight: float = 0.10,
    ) -> torch.Tensor:
        """Compute closed-form Harmonic Potential Discrepancy in O(D * rank) flops."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        energy, curvature, _ = self.compute_energy_and_curvature(z_ctx, z_tgt)

        # Base score is the potential energy well height (Mahalanobis spring displacement)
        base_score = torch.sqrt(torch.clamp(energy, min=1e-8))

        if include_curvature:
            # Normalized curvature using running statistics
            std_curv = torch.sqrt(torch.clamp(self.curv_var, min=1e-6))
            norm_curv = (curvature - self.curv_mean) / std_curv
            score = base_score + curvature_weight * F.relu(norm_curv)
        else:
            score = base_score

        return score

    @torch.no_grad()
    def fit_mahalanobis_covariance(
        self,
        context_windows,
        target_windows,
        batch_size: int = 512,
        reg: float = 1e-3,
    ) -> None:
        """Fit empirical curvature statistics and residual covariance on nominal data."""
        self.eval()
        device = next(self.parameters()).device
        
        # Collect nominal curvatures to calibrate running mean and variance
        all_curvatures = []
        N = len(context_windows)
        for i in range(0, N, batch_size):
            chunk_ctx = context_windows[i : i + batch_size]
            chunk_tgt = target_windows[i : i + batch_size]
            if isinstance(chunk_ctx, np.ndarray):
                ctx_b = torch.from_numpy(chunk_ctx).float().to(device)
                tgt_b = torch.from_numpy(chunk_tgt).float().to(device)
            else:
                ctx_b = chunk_ctx.float().to(device)
                tgt_b = chunk_tgt.float().to(device)
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            _, curvature, _ = self.compute_energy_and_curvature(z_ctx, z_tgt)
            all_curvatures.append(curvature.cpu())

        curv_tensor = torch.cat(all_curvatures, dim=0)
        self.curv_mean.copy_(torch.mean(curv_tensor))
        self.curv_var.copy_(torch.var(curv_tensor, unbiased=False))
        self.curv_count.copy_(torch.tensor(float(len(curv_tensor))))


        # Fit residual mean & precision for compatibility
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            mu, _, _ = self.spring_head(z_ctx)
            return z_tgt - mu

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


HarmonicSpringJEPA = HarmonicSpringJEPAModel
SpringJEPA = HarmonicSpringJEPAModel
