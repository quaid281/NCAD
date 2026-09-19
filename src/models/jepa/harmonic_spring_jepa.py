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
from src.models.geometric_layers import ResolventPurification
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

        # Resolvent purification for Riemannian metric operator (Chapter 6, §4.2)
        self.purification = ResolventPurification(dim=latent_dim, gamma=eps)

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
    4. Resolvent-purified Energy and Curvature evaluation without ODE integration or autograd.
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

        # Hankel Moment Filter for algebraic trajectory consistency (Ten Proofs, Ch. 7)
        from src.models.geometric_layers import HankelMomentFilter
        self.hankel_filter = HankelMomentFilter(latent_dim=latent_dim, hankel_order=4)

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
        """Compute resolvent-purified potential energy Phi, curvature Laplacian Tr(M), and log det."""
        B = z_ctx.size(0)
        mu, L, d = self.spring_head(z_ctx)

        # 1. Coordinate error e = z_tgt - mu
        e = z_tgt - mu  # (B, D)

        # Reconstruct Stiffness Matrix M = L L^T + diag(d) with Resolvent Purification
        diag_d = torch.diag_embed(d)  # (B, D, D)
        M_raw = torch.bmm(L, L.transpose(1, 2)) + diag_d  # (B, D, D)
        M_purified = self.spring_head.purification(M_raw, is_psd_matrix=True)

        # 2. Resolvent-purified Potential Energy: e^T M_purified e
        energy = torch.sum((e.unsqueeze(1) @ M_purified).squeeze(1) * e, dim=-1)  # (B,)

        # 3. Exact Laplacian Curvature: Tr(M_purified)
        curvature = torch.diagonal(M_purified, dim1=-2, dim2=-1).sum(dim=-1)  # (B,)

        # 4. Exact Log-Determinant via Matrix Determinant Lemma:
        # det(diag(d) + L L^T) = det(I_r + L^T diag(d)^{-1} L) * prod(d_j)
        log_det_diag = torch.sum(torch.log(d), dim=-1)  # (B,)
        L_scaled = L / torch.sqrt(d.unsqueeze(-1))  # (B, D, r)
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
        """Forward pass for harmonic spring predictive learning with Hankel filtering."""
        z_ctx = self.hankel_filter(self.context_encoder(context_windows))

        if target_windows is None:
            return z_ctx, None, None, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt_true = self.hankel_filter(self.target_encoder(target_windows))

        energy, curvature, log_det = self.compute_energy_and_curvature(z_ctx, z_tgt_true)

        return z_ctx, z_tgt_true, energy, curvature, log_det

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        det_weight: float = 0.0,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute streamlined pure Harmonic Spring potential energy objective."""
        z_ctx, z_tgt_true, energy, curvature, log_det = self.forward(ctx, tgt)

        # Pure Harmonic Energy Minimization: E[Phi(z_tgt | z_ctx)]
        loss_energy = torch.mean(energy)
        total_loss = loss_energy

        with torch.no_grad():
            loss_det = -torch.mean(log_det) / float(self.latent_dim)

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_energy": float(loss_energy.item()),
            "loss_det": float(loss_det.item()),
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
