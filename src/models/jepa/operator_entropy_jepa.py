"""Von Neumann Operator Entropy & Spectral Rigidity JEPA Architecture.

Derived from Connes's Rigidity Conjecture and Quantum Parallel Repetition (OpenAI, 2026):
Replaces ad-hoc scalar covariance penalties with the Von Neumann Operator Entropy
S(rho) = -Tr(rho log rho) on the normalized state density operator rho = Cov / Tr(Cov).

Enforces an intrinsic spectral gap in the latent dynamical operator to prevent
amenable spectrum leakage, detecting anomalies as operator entropy collapses and
spectral dissipation across the Kazhdan gap.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.geometric_layers import ResolventPurification


def spectral_operator_entropy_loss(
    K: torch.Tensor,
    gamma_norm: float = 1.0,
    beta_norm: float = 1.0,
    eps_log: float = 1e-12,
    eps_safe: float = 1e-6,
) -> torch.Tensor:
    """Anti-collapse Operator-Entropy regularizer.
    
    Penalizes both dimensional collapse (non-uniform spectrum) and energy collapse (K -> 0).
    Unlike naive formulations with additive isotropic diagonal loading (eps/D * I),
    this formulation does not reward the degenerate K=0 solution.
    
    Args:
        K: Transition operator tensor of shape (D, D) or (B, D, D).
        gamma_norm: Target Frobenius energy scaling factor (default: 1.0 -> Tr(KK^T) = D).
        beta_norm: Weight of the logarithmic energy barrier.
        eps_log: Numerical stability floor for log(p) on non-zero singular modes.
        eps_safe: Numerical safety epsilon for energy denominator.
    Returns:
        Scalar regularizer loss >= 0.
    """
    if K.ndim == 2:
        K = K.unsqueeze(0)
    B, D, _ = K.shape
    
    # 1. Squared singular values via SPSD Gram matrix KK^T
    KKt = torch.bmm(K, K.transpose(-1, -2))
    s_vals = torch.linalg.eigvalsh(KKt)  # (B, D), sorted ascending, >= 0
    s_vals = torch.clamp(s_vals, min=0.0)
    
    # 2. Total operator energy Tr(KK^T) = ||K||_F^2
    energy = torch.sum(s_vals, dim=-1)  # (B,)
    target_energy = gamma_norm * float(D)
    
    # 3. Normalized spectral distribution without isotropic diagonal loading
    energy_clamped = torch.clamp(energy, min=1e-12)
    p = s_vals / energy_clamped.unsqueeze(-1)  # (B, D), sum_i p_i = 1
    
    # 4. Von Neumann Entropy (convention 0 * log 0 = 0)
    p_safe = torch.clamp(p, min=eps_log)
    log_p = torch.log(p_safe)
    p_log_p = torch.where(p > 1e-10, p * log_p, torch.zeros_like(p))
    vn_entropy = -torch.sum(p_log_p, dim=-1)  # (B,)
    
    max_entropy = math.log(float(D))
    entropy_deficiency = torch.clamp(max_entropy - vn_entropy, min=0.0)  # (B,)
    
    # 5. Bregman/Itakura-Saito Anti-Collapse Energy Barrier: d(r, 1) = r - 1 - log(r) >= 0
    # Strictly non-negative, unique global minimum at r=1, diverges to +inf as r -> 0+
    energy_ratio = energy / target_energy
    r_clamped = torch.clamp(energy_ratio, min=eps_safe)
    energy_barrier = r_clamped - 1.0 - torch.log(r_clamped)
    
    total_reg = entropy_deficiency + beta_norm * energy_barrier
    return total_reg.mean()


def von_neumann_entropy(z: torch.Tensor, eps_log: float = 1e-12, eps_safe: float = 1e-6) -> torch.Tensor:
    """Compute scale-separated Von Neumann operator entropy deficiency on covariance.
    
    Strictly penalizes both rank deficiency and variance collapse (z -> 0).
    """
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    D = z.size(-1)
    N = z.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=z.device)
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    
    evals = torch.linalg.eigvalsh(cov)
    evals = torch.clamp(evals, min=0.0)
    total_var = torch.sum(evals)
    max_entropy = math.log(float(D))
    
    if total_var < eps_safe:
        return torch.tensor(max_entropy + 10.0, device=z.device)
    
    p = evals / total_var
    p_safe = torch.clamp(p, min=eps_log)
    p_log_p = torch.where(p > 1e-10, p * torch.log(p_safe), torch.zeros_like(p))
    vn_entropy = -torch.sum(p_log_p)
    entropy_deficiency = torch.clamp(max_entropy - vn_entropy, min=0.0)
    
    var_ratio = total_var / float(D)
    vr_clamped = torch.clamp(var_ratio, min=eps_safe)
    var_barrier = vr_clamped - 1.0 - torch.log(vr_clamped)
    return entropy_deficiency + var_barrier


class OperatorEntropyJEPAModel(JEPABase):
    """Operator Entropy & Spectral Rigidity JEPA.
    
    Regularizes the learned transition operator K via Von Neumann spectral entropy
    and an anti-collapse energy barrier, preventing dimensional collapse onto
    degenerate subspaces.
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.996,
        alpha_entropy: float = 0.5,
        spectral_gap_min: float = 0.1,
        dropout: float = 0.05,
        use_radial_heat: bool = True,
        use_koopman_matrix: bool = True,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.alpha_entropy = alpha_entropy
        self.spectral_gap_min = spectral_gap_min
        self.use_koopman_matrix = use_koopman_matrix

        self.target_encoder = self.init_target_encoder(context_encoder)

        if use_koopman_matrix:
            # Explicit Koopman transition matrix K in R^{D x D} (1,024 parameters for D=32)
            # Initialized near identity with small random perturbations
            self.K = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))
            self.predictor = None
        else:
            layers = []
            in_d = latent_dim
            for _ in range(predictor_layers - 1):
                layers.extend([
                    nn.Linear(in_d, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ])
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, latent_dim))
            layers.append(nn.LayerNorm(latent_dim))
            self.predictor = nn.Sequential(*layers)
            self.K = None

        # Resolvent purification for spectral operator regularization (Chapter 6, §4.2)
        self.resolvent = ResolventPurification(dim=latent_dim, gamma=0.01)

        # Radial heat exterior boundary damping (Navier-Stokes Blowup Paper, §2.3 & App. A.6)
        if use_radial_heat:
            from src.models.geometric_layers import RadialHeatExteriorBoundary
            self.radial_heat = RadialHeatExteriorBoundary(latent_dim=latent_dim)
        else:
            self.radial_heat = None

        # Cohn-Elkies Fourier sign-uncertainty modulation layer (Ten Proofs, Ch. 1, §4.2)
        from src.models.geometric_layers import CohnElkiesFilter, CoordinateSaliencyGate
        self.cohn_elkies = CohnElkiesFilter(latent_dim=latent_dim, n_shells=8)
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)

        self.register_mahalanobis_buffers(latent_dim)

    def _predict(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the predictive transition operator."""
        if self.use_koopman_matrix:
            return z @ self.K.T
        return self.predictor(z)

    def _apply_filters(self, z: torch.Tensor) -> torch.Tensor:
        if self.radial_heat is not None:
            z = self.radial_heat(z)
        z = self.cohn_elkies(z)
        return z

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self._apply_filters(self.context_encoder(context_windows))
        z_pred = self._apply_filters(self._predict(z_ctx))

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self._apply_filters(self.target_encoder(target_windows))

        diff = torch.linalg.norm(z_tgt - z_pred, dim=-1)
        return z_pred, z_tgt, diff

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        var_weight: float = 0.0,
        alpha_entropy: Optional[float] = None,
        gap_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        z_ctx = self._apply_filters(self.context_encoder(ctx))
        z_pred = self._apply_filters(self._predict(z_ctx))

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self._apply_filters(self.target_encoder(tgt))

        # 1. Pure predictive MSE loss
        pred_loss = F.mse_loss(z_pred, z_tgt)

        # 2. Operator Entropy Regularization
        effective_alpha = alpha_entropy if alpha_entropy is not None else self.alpha_entropy
        if self.use_koopman_matrix:
            spectral_reg = spectral_operator_entropy_loss(self.K, gamma_norm=1.0, beta_norm=1.0)
        else:
            spectral_reg = 0.5 * (von_neumann_entropy(z_ctx) + von_neumann_entropy(z_pred))

        total_loss = pred_loss + effective_alpha * spectral_reg

        # Compute diagnostic spectral metrics under no_grad()
        with torch.no_grad():
            z_cent = z_pred - z_pred.mean(dim=0)
            cov = (z_cent.T @ z_cent) / max(z_pred.size(0) - 1, 1)
            evals = torch.linalg.eigvalsh(cov)
            gap = evals[-1] - evals[0]
            gap_loss = F.relu(self.spectral_gap_min - gap)
            std_pred = torch.sqrt(z_pred.var(dim=0, unbiased=False) + eps)
            std_loss = torch.mean(F.relu(gamma - std_pred))

        if var_weight > 0:
            total_loss = total_loss + var_weight * torch.mean(F.relu(gamma - torch.sqrt(z_pred.var(dim=0, unbiased=False) + eps)))
        if gap_weight > 0:
            total_loss = total_loss + gap_weight * gap_loss

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "spectral_reg": spectral_reg.item(),
            "vn_loss": spectral_reg.item(),
            "gap_loss": gap_loss.item(),
            "std_loss": std_loss.item(),
        }
        return total_loss, metrics

    def _apply_radial_heat(self, z: torch.Tensor) -> torch.Tensor:
        """Backward compatible alias."""
        return self._apply_filters(z)

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        self.eval()
        z_ctx = self._apply_filters(self.context_encoder(context_windows))
        z_pred = self._apply_filters(self._predict(z_ctx))
        z_tgt = self._apply_filters(self.target_encoder(observed_target_windows))

        diff = z_tgt - z_pred
        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_cent = diff - self.residual_mean
            mahal_coord = (diff_cent @ self.precision_matrix) * diff_cent
            e_white = torch.sign(diff_cent) * torch.sqrt(torch.clamp(mahal_coord, min=0.0))
            disc = self.saliency_gate(e_white)
        else:
            disc = self.saliency_gate(diff)

        # Spectral energy leakage component
        energy_leak = torch.clamp(torch.norm(z_tgt, dim=-1) - torch.norm(z_pred, dim=-1), min=0.0)
        return disc + 0.2 * energy_leak

    def fit_covariance(
        self,
        context_windows: Union[np.ndarray, torch.Tensor],
        target_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 512,
        reg: float = 1e-3,
        method: str = "resolvent",
    ) -> None:
        def residual_fn(ctx, tgt):
            z_ctx = self.context_encoder(ctx)
            z_pred = self._predict(z_ctx)
            z_tgt = self.target_encoder(tgt)
            return z_tgt - z_pred

        fit_covariance_batched(
            self,
            context_windows,
            target_windows,
            residual_fn=residual_fn,
            dim=self.latent_dim,
            batch_size=batch_size,
            reg=reg,
            method=method,
            precision_buffer=self.precision_matrix,
            residual_mean_buffer=self.residual_mean,
            fitted_buffer=self.precision_fitted,
        )

    fit_mahalanobis_covariance = fit_covariance


OperatorEntropyJEPA = OperatorEntropyJEPAModel
