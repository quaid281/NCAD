"""Hamiltonian Symplectic Joint Embedding Predictive Architecture (HamiltonianSymplecticJEPA).

Rooted in Classical Mechanics and Symplectic Geometry:
1. Decomposes latent state z in R^D into generalized coordinates (q, p) in R^{D/2} x R^{D/2}:
       q: generalized position (static/spatial configuration)
       p: generalized momentum (kinetic/temporal rate of change)
2. Learned Hamiltonian Energy Function H(q, p) = 1/2 * ||p||^2 + V_theta(q).
3. Symplectic Leapfrog Integrator rolls forward (q_ctx, p_ctx) -> (q_hat_tgt, p_hat_tgt)
   under exact phase space volume preservation.
4. Dual Physical Anomaly Scoring:
       State Residual (Point Anomaly):                S_state  = ||q_tgt - q_hat_tgt||_2 + ||p_tgt - p_hat_tgt||_2
       Energy Conservation Violation (Contextual):    S_energy = |H(q_tgt, p_tgt) - H(q_ctx, p_ctx)|
5. Total Anomaly Score:
       S = S_state + beta * S_energy
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class PotentialEnergyNet(nn.Module):
    """Learned scalar potential energy field V_theta: R^{D/2} -> R."""

    def __init__(self, coord_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.coord_dim = coord_dim
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute potential energy V(q), shape (B,)."""
        return self.net(q).squeeze(-1)

    def grad_V(self, q: torch.Tensor) -> torch.Tensor:
        """Compute conservative force -nabla_q V(q) via autograd or analytic forward pass."""
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            V = self.forward(q_in)
            grad = torch.autograd.grad(V.sum(), q_in, create_graph=True)[0]
        return grad


class HamiltonianSymplecticJEPAModel(JEPABase):
    """Hamiltonian Symplectic JEPA (HamiltonianSymplecticJEPA)."""

    def __init__(
        self,
        context_encoder: nn.Module,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        step_size: float = 0.1,
        num_leapfrog_steps: int = 3,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, f"latent_dim ({latent_dim}) must be even for (q, p) split"
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.coord_dim = latent_dim // 2
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.ema_decay = ema_decay

        self.target_encoder = self.init_target_encoder(context_encoder)

        self.potential_net = PotentialEnergyNet(
            coord_dim=self.coord_dim,
            hidden_dim=hidden_dim,
        )

        self.register_mahalanobis_buffers(latent_dim)

    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Total energy H(q, p) = 1/2 * ||p||^2 + V(q), shape (B,)."""
        kinetic = 0.5 * torch.sum(p ** 2, dim=-1)
        potential = self.potential_net(q)
        return kinetic + potential

    def leapfrog_step(self, q: torch.Tensor, p: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform 1-step symplectic leapfrog integration."""
        # p_{1/2} = p_0 - eps/2 * grad_V(q_0)
        grad_0 = self.potential_net.grad_V(q)
        p_half = p - 0.5 * eps * grad_0

        # q_1 = q_0 + eps * p_{1/2}
        q_next = q + eps * p_half

        # p_1 = p_{1/2} - eps/2 * grad_V(q_1)
        grad_1 = self.potential_net.grad_V(q_next)
        p_next = p_half - 0.5 * eps * grad_1

        return q_next, p_next

    def symplectic_rollout(self, q_0: torch.Tensor, p_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unroll Hamiltonian dynamics forward for K leapfrog steps."""
        q, p = q_0, p_0
        for _ in range(self.num_leapfrog_steps):
            q, p = self.leapfrog_step(q, p, self.step_size)
        return q, p

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        z_ctx = self.context_encoder(context_windows)
        q_ctx = z_ctx[:, : self.coord_dim]
        p_ctx = z_ctx[:, self.coord_dim :]

        q_hat_tgt, p_hat_tgt = self.symplectic_rollout(q_ctx, p_ctx)
        z_pred = torch.cat([q_hat_tgt, p_hat_tgt], dim=-1)

        if target_windows is None:
            return z_ctx, None, z_pred, q_hat_tgt

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self.target_encoder(target_windows)

        return z_ctx, z_tgt, z_pred, q_hat_tgt

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        state_weight: float = 1.0,
        energy_weight: float = 0.5,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Symplectic state loss + Energy conservation loss + VICReg."""
        z_ctx, z_tgt, z_pred, _ = self.forward(ctx, tgt)

        q_ctx = z_ctx[:, : self.coord_dim]
        p_ctx = z_ctx[:, self.coord_dim :]
        q_tgt = z_tgt[:, : self.coord_dim]
        p_tgt = z_tgt[:, self.coord_dim :]

        # 1. State Trajectory Residual Loss: ||z_tgt - z_pred||^2
        loss_state = torch.mean(torch.sum((z_tgt - z_pred) ** 2, dim=-1))

        # 2. Hamiltonian Energy Conservation Loss: |H(tgt) - H(ctx)|^2
        H_ctx = self.compute_hamiltonian(q_ctx, p_ctx)
        H_tgt = self.compute_hamiltonian(q_tgt, p_tgt)
        loss_energy = torch.mean((H_tgt - H_ctx) ** 2)

        # 3. Representation Non-Collapse (VICReg)
        std_c = torch.sqrt(torch.var(z_ctx, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_ctx, eps=eps)

        total_loss = (
            state_weight * loss_state
            + energy_weight * loss_energy
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_state": float(loss_state.item()),
            "loss_energy": float(loss_energy.item()),
            "mean_energy_ctx": float(torch.mean(H_ctx).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        beta_energy: float = 0.50,
        **kwargs,
    ) -> torch.Tensor:
        """Compute decoupled dual anomaly score: State Residual + Energy Conservation Deficit."""
        self.eval()
        z_ctx = self.context_encoder(context_windows)
        z_tgt = self.target_encoder(observed_target_windows)

        q_ctx = z_ctx[:, : self.coord_dim]
        p_ctx = z_ctx[:, self.coord_dim :]
        q_tgt = z_tgt[:, : self.coord_dim]
        p_tgt = z_tgt[:, self.coord_dim :]

        q_hat_tgt, p_hat_tgt = self.symplectic_rollout(q_ctx, p_ctx)
        z_pred = torch.cat([q_hat_tgt, p_hat_tgt], dim=-1)

        # 1. State Residual (Point Anomaly)
        score_state = torch.sqrt(torch.sum((z_tgt - z_pred) ** 2, dim=-1) + 1e-8)  # (B,)

        # 2. Energy Conservation Violation (Contextual Anomaly)
        H_ctx = self.compute_hamiltonian(q_ctx, p_ctx)
        H_tgt = self.compute_hamiltonian(q_tgt, p_tgt)
        score_energy = torch.abs(H_tgt - H_ctx)                                     # (B,)

        total_score = score_state + beta_energy * score_energy
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            z_ctx = self.context_encoder(ctx_b)
            z_tgt = self.target_encoder(tgt_b)
            q_ctx = z_ctx[:, : self.coord_dim]
            p_ctx = z_ctx[:, self.coord_dim :]
            q_hat_tgt, p_hat_tgt = self.symplectic_rollout(q_ctx, p_ctx)
            z_pred = torch.cat([q_hat_tgt, p_hat_tgt], dim=-1)
            return z_tgt - z_pred

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


HamiltonianSymplecticJEPA = HamiltonianSymplecticJEPAModel
