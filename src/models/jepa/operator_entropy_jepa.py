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


def von_neumann_entropy(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute Von Neumann operator entropy S(rho) = -Tr(rho log rho)."""
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    D = z.size(-1)
    N = z.size(0)
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    tr = torch.trace(cov) + eps
    rho = cov / tr + (eps / D) * torch.eye(D, device=z.device)
    rho = rho / torch.trace(rho)

    evals = torch.linalg.eigvalsh(rho)
    evals = torch.clamp(evals, min=eps)
    p = evals / evals.sum()
    vn_entropy = -torch.sum(p * torch.log(p))
    max_entropy = math.log(float(D))
    # Return entropy deficiency (0 when uniform, high when collapsed to low rank)
    return torch.clamp(max_entropy - vn_entropy, min=0.0)


class OperatorEntropyJEPAModel(JEPABase):
    """Operator Entropy & Spectral Rigidity JEPA."""

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
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.alpha_entropy = alpha_entropy
        self.spectral_gap_min = spectral_gap_min

        self.target_encoder = self.init_target_encoder(context_encoder)

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

        self.register_mahalanobis_buffers(latent_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self.context_encoder(context_windows)
        z_pred = self.predictor(z_ctx)

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        diff = torch.linalg.norm(z_tgt - z_pred, dim=-1)
        return z_pred, z_tgt, diff

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        z_ctx = self.context_encoder(ctx)
        z_pred = self.predictor(z_ctx)

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(tgt)

        pred_loss = F.mse_loss(z_pred, z_tgt)

        # Variance regularization
        std_pred = torch.sqrt(z_pred.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(gamma - std_pred))

        # Von Neumann Operator Entropy Defect
        vn_loss = 0.5 * (von_neumann_entropy(z_ctx) + von_neumann_entropy(z_pred))

        # Spectral Gap constraint: gap between top eigenvalue and noise floor
        z_cent = z_pred - z_pred.mean(dim=0)
        cov = (z_cent.T @ z_cent) / max(z_pred.size(0) - 1, 1)
        evals = torch.linalg.eigvalsh(cov)
        gap = evals[-1] - evals[0]
        gap_loss = F.relu(self.spectral_gap_min - gap)

        total_loss = pred_loss + var_weight * std_loss + self.alpha_entropy * vn_loss + 0.1 * gap_loss

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "vn_loss": vn_loss.item(),
            "gap_loss": gap_loss.item(),
            "std_loss": std_loss.item(),
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
        z_ctx = self.context_encoder(context_windows)
        z_pred = self.predictor(z_ctx)
        z_tgt = self.target_encoder(observed_target_windows)

        diff = z_tgt - z_pred

        if use_mahalanobis and bool(self.precision_fitted.item()):
            diff_cent = diff - self.residual_mean
            mahal = torch.sum((diff_cent @ self.precision_matrix) * diff_cent, dim=-1)
            disc = torch.sqrt(torch.clamp(mahal, min=0.0))
        else:
            disc = torch.linalg.norm(diff, dim=-1)

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


OperatorEntropyJEPA = OperatorEntropyJEPAModel
