"""Navier-Stokes Reynolds Stress Closure JEPA Architecture.

Derived from the finite-time Navier-Stokes singularity construction (OpenAI, 2026):
Decomposes latent dynamics into a macroscopic background flow z_bar and high-frequency
micro-fluctuations across channels/patches. The quadratic Reynolds stress tensor
Sigma = 1/C sum_c (z'_c (x) z'_c) generates an internal momentum flux that balances
convective shear.

Anomalies are detected as breakdowns of the hydrodynamic stress closure balance
and departures from the Admissible Stress Cone S_+.
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


class ReynoldsStressClosureHead(nn.Module):
    """Predicts the target Reynolds stress tensor Sigma in S_+ from macro context
    via Admissible Stress Cone Projection (Navier-Stokes Blowup Paper, §4.3 & App. C)
    and Resolvent Purification.
    """

    def __init__(
        self,
        latent_dim: int,
        stress_dim: int = 16,
        gamma: float = 1e-2,
        use_cone_proj: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.stress_dim = stress_dim
        self.use_cone_proj = use_cone_proj
        if use_cone_proj:
            from src.models.geometric_layers import AdmissibleStressConeProjection
            self.cone_proj = AdmissibleStressConeProjection(
                latent_dim=latent_dim,
                stress_dim=stress_dim,
                gamma=gamma,
            )
            self.net = None
            self.purification = None
        else:
            self.cone_proj = None
            self.net = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.SiLU(),
                nn.Linear(latent_dim * 2, stress_dim * stress_dim),
            )
            self.purification = ResolventPurification(dim=stress_dim, gamma=gamma)

    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        if z_macro.ndim == 3:
            z_macro = z_macro.mean(dim=1)
        if self.cone_proj is not None:
            return self.cone_proj(z_macro)
        raw = self.net(z_macro)
        mat = raw.reshape(-1, self.stress_dim, self.stress_dim)
        return self.purification(mat)


class ReynoldsStressJEPAModel(JEPABase):
    """Reynolds-Stress Navier-Stokes JEPA with Admissible Cone Constraint."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        stress_dim: int = 16,
        hidden_dim: int = 64,
        predictor_layers: int = 2,
        ema_decay: float = 0.996,
        alpha_stress: float = 0.5,
        alpha_cone: float = 0.2,
        dropout: float = 0.05,
        use_shearing_pulses: bool = True,
        use_admissible_cone: bool = True,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.stress_dim = min(stress_dim, latent_dim)
        self.ema_decay = ema_decay
        self.alpha_stress = alpha_stress
        self.alpha_cone = alpha_cone

        self.target_encoder = self.init_target_encoder(context_encoder)

        # Shearing Wavelet Pulse Block (Navier-Stokes Blowup Paper, §7)
        if use_shearing_pulses:
            from src.models.geometric_layers import ShearingWaveletPulseBlock
            self.shearing_pulse = ShearingWaveletPulseBlock(channels=latent_dim)
        else:
            self.shearing_pulse = None

        # Hankel Moment Filter for algebraic trajectory consistency (Ten Proofs, Ch. 7)
        from src.models.geometric_layers import CohnElkiesFilter, HankelMomentFilter
        self.hankel_filter = HankelMomentFilter(latent_dim=latent_dim, hankel_order=4)
        self.cohn_elkies = CohnElkiesFilter(latent_dim=latent_dim, n_shells=8)

        # Standard latent predictor
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
        self.predictor = nn.Sequential(*layers)

        # Macro-to-stress projection with Admissible Stress Cone
        self.stress_head = ReynoldsStressClosureHead(
            latent_dim=latent_dim,
            stress_dim=self.stress_dim,
            use_cone_proj=use_admissible_cone,
        )

        # Stress compression if latent_dim != stress_dim
        if self.latent_dim != self.stress_dim:
            self.stress_proj = nn.Linear(latent_dim, self.stress_dim, bias=False)
        else:
            self.stress_proj = nn.Identity()

        self.register_mahalanobis_buffers(latent_dim)

    def compute_observed_stress(self, z_tgt: torch.Tensor) -> torch.Tensor:
        """Compute the empirical quadratic Reynolds stress tensor Sigma_obs in S_+."""
        if z_tgt.ndim == 3:
            z_tgt = z_tgt.mean(dim=1)
        z_s = self.stress_proj(z_tgt)  # [B, d_s]
        # Outer product: [B, d_s, d_s]
        sigma = torch.bmm(z_s.unsqueeze(2), z_s.unsqueeze(1))
        # Ensure symmetric positive semi-definiteness with minimum isotropic floor
        eye = torch.eye(self.stress_dim, device=z_tgt.device, dtype=z_tgt.dtype).unsqueeze(0)
        return sigma + 1e-3 * eye

    def _extract_context(self, context_windows: torch.Tensor) -> torch.Tensor:
        """Extract context representations with Hankel filtering and shearing wavelet pulses."""
        z_ctx = self.context_encoder(context_windows)
        if self.shearing_pulse is not None:
            if z_ctx.ndim == 3:
                z_ctx = self.shearing_pulse(z_ctx)
            elif z_ctx.ndim == 2:
                z_ctx = self.shearing_pulse(z_ctx.unsqueeze(1)).squeeze(1)
        # Apply Bounded-Moment Hankel filter
        z_ctx = self.hankel_filter(z_ctx)
        return z_ctx

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self._extract_context(context_windows)
        z_pred = self.cohn_elkies(self.predictor(z_ctx))

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.cohn_elkies(self.hankel_filter(self.target_encoder(target_windows)))

        sigma_obs = self.compute_observed_stress(z_tgt)
        sigma_pred = self.stress_head(z_ctx)
        stress_diff = torch.linalg.norm(sigma_obs - sigma_pred, dim=(-2, -1))
        return z_pred, z_tgt, stress_diff

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        cov_weight: float = 0.0,
        var_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the streamlined, pure predictive MSE training objective."""
        z_ctx = self._extract_context(ctx)
        z_pred = self.cohn_elkies(self.predictor(z_ctx))

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.cohn_elkies(self.hankel_filter(self.target_encoder(tgt)))

        # Pure predictive alignment: ||z_pred - z_tgt||^2
        pred_loss = F.mse_loss(z_pred, z_tgt)
        total_loss = pred_loss

        with torch.no_grad():
            sigma_obs = self.compute_observed_stress(z_tgt)
            sigma_pred = self.stress_head(z_ctx)
            stress_loss = F.mse_loss(sigma_pred, sigma_obs)

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "stress_loss": stress_loss.item(),
            "cone_defect": 0.0,
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        self.eval()
        z_ctx = self._extract_context(context_windows)
        z_pred = self.predictor(z_ctx)
        z_tgt = self.target_encoder(observed_target_windows)

        diff = z_tgt - z_pred

        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_cent = diff - self.residual_mean
            mahal = torch.sum((diff_cent @ self.precision_matrix) * diff_cent, dim=-1)
            base_disc = torch.sqrt(torch.clamp(mahal, min=0.0))
        else:
            base_disc = torch.linalg.norm(diff, dim=-1)

        # Reynolds Stress Closure Discrepancy
        sigma_obs = self.compute_observed_stress(z_tgt)
        sigma_pred = self.stress_head(z_ctx)
        stress_disc = torch.linalg.norm(sigma_obs - sigma_pred, dim=(-2, -1))

        # Combined hydro-kinetic anomaly score
        return base_disc + 0.5 * stress_disc

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
            z_pred = self.predictor(z_ctx)
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


ReynoldsStressJEPA = ReynoldsStressJEPAModel
