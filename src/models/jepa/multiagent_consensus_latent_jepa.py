"""Multi-Agent Consensus Latent World Model JEPA (MultiAgentConsensusLatentJEPA).

Synthesized from Interlat (ACL 2026: "Direct Inter-Agent Continuous Latent Communication"):
1. Multi-Agent Physical Decomposition:
       Each sensor channel c in {1..C} is treated as an autonomous physical agent.
       z_t^c = E_agent(x_t^c) in R^{d_agent}
       Z_t = [z_t^1, z_t^2, ..., z_t^C] in R^{C x d_agent}
2. Continuous Latent Inter-Agent Communication Bus:
       M_t = AgentMultiheadAttention(query=Z_t, key=Z_t, value=Z_t) in R^{C x d_agent}
       Continuous latent message passing without discrete serialization or tokenization.
3. Coupled Autonomous Agent Rollout:
       h_{t+1}^c = AgentGRUCell([z_t^c, m_t^c], h_t^c)
       z_hat_{t+1}^c = z_t^c + Readout(h_{t+1}^c)
4. Consensus Dissonance Anomaly Detection & Root Cause Localization:
       S_local      = sum_c ||z_tgt^c - z_hat^c||_2 (individual sensor error)
       S_dissonance = sum_c ||z_hat^c - m_t^c||_2   (peer physical law violation)
       S_total      = S_local + lambda_cons * S_dissonance
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched
from src.models.jepa.flow_ts_jepa import von_neumann_operator_entropy_loss


class AgentFrameEncoder(nn.Module):
    """Encodes each channel c independently: x_t^c in R^1 -> z_t^c in R^{d_agent}."""

    def __init__(self, d_agent: int = 16, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.d_agent = d_agent
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_agent),
            nn.LayerNorm(d_agent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, T, C). Returns: (B, T, C, d_agent)."""
        B, T, C = x.shape
        x_unflat = x.unsqueeze(-1)  # (B, T, C, 1)
        x_flat = x_unflat.contiguous().view(B * T * C, 1)
        z_flat = self.net(x_flat)
        return z_flat.view(B, T, C, self.d_agent)


class ContinuousCommunicationBus(nn.Module):
    """Continuous latent communication bus where agents exchange physical messages."""

    def __init__(self, d_agent: int = 16, num_heads: int = 2, dropout: float = 0.05):
        super().__init__()
        self.d_agent = d_agent
        # Cross-agent attention along the channel dimension C
        self.attn = nn.MultiheadAttention(
            embed_dim=d_agent,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_agent)

    def forward(self, z_agents: torch.Tensor) -> torch.Tensor:
        """Args: z_agents of shape (B, C, d_agent). Returns: messages of shape (B, C, d_agent)."""
        z_norm = self.norm(z_agents)
        messages, _ = self.attn(z_norm, z_norm, z_norm)
        return messages


class AgentRecurrentCore(nn.Module):
    """Recurrent transition core for a single agent receiving local state + peer messages."""

    def __init__(self, d_agent: int = 16, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.d_agent = d_agent
        self.hidden_dim = hidden_dim
        # Input: own state (d_agent) + consensus message from peers (d_agent)
        self.cell = nn.GRUCell(d_agent * 2, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_agent),
        )

    def step(
        self,
        z_t: torch.Tensor,
        m_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: z_t (B*C, d_agent), m_t (B*C, d_agent), h_prev (B*C, hidden_dim)."""
        inp = torch.cat([z_t, m_t], dim=-1)
        h_next = self.cell(inp, h_prev)
        delta = self.readout(h_next)
        z_next = z_t + delta
        return z_next, h_next


class MultiAgentConsensusLatentJEPAModel(JEPABase):
    """Multi-Agent Consensus Latent World Model JEPA (Interlat ACL 2026 insight)."""

    def __init__(
        self,
        input_dim: int = 1,
        d_agent: int = 16,
        hidden_dim: int = 32,
        num_heads: int = 2,
        ema_decay: float = 0.996,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_agent = d_agent
        self.hidden_dim = hidden_dim
        self.latent_dim = input_dim * d_agent
        self.ema_decay = ema_decay

        self.context_encoder = AgentFrameEncoder(
            d_agent=d_agent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_encoder = self.init_target_encoder(self.context_encoder)

        self.comm_bus = ContinuousCommunicationBus(
            d_agent=d_agent,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.agent_core = AgentRecurrentCore(
            d_agent=d_agent,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_mahalanobis_buffers(self.latent_dim)

    def forward_latent_trajectory(
        self,
        context_windows: torch.Tensor,
        horizon: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracks context, passes continuous latent messages, and unrolls agent swarms."""
        B, C_len, C = context_windows.shape
        Z_ctx_agents = self.context_encoder(context_windows)  # (B, C_len, C, d_agent)

        h = torch.zeros(B * C, self.hidden_dim, device=context_windows.device)
        Z_ctx_pred = []

        # Phase 1: Context Tracking with Inter-Agent Communication
        for t in range(C_len):
            z_t = Z_ctx_agents[:, t]  # (B, C, d_agent)
            m_t = self.comm_bus(z_t)  # (B, C, d_agent)

            z_t_flat = z_t.contiguous().view(B * C, self.d_agent)
            m_t_flat = m_t.contiguous().view(B * C, self.d_agent)

            z_next_flat, h = self.agent_core.step(z_t_flat, m_t_flat, h)
            z_next = z_next_flat.view(B, C, self.d_agent)
            Z_ctx_pred.append(z_next.view(B, -1))  # (B, C*d_agent)

        Z_ctx_pred = torch.stack(Z_ctx_pred, dim=1)  # (B, C_len, D)

        # Phase 2: Autonomous Suspect Rollout (Closed-loop multi-agent dialogue)
        curr_z = Z_ctx_pred[:, -1].view(B, C, self.d_agent)
        Z_sus_pred = []
        dissonances = []

        for tau in range(horizon):
            m_tau = self.comm_bus(curr_z)  # (B, C, d_agent)
            # Dissonance: distance between agent trajectory and consensus message
            diss = torch.mean(torch.norm(curr_z - m_tau, dim=-1), dim=-1)  # (B,)
            dissonances.append(diss)

            Z_sus_pred.append(curr_z.view(B, -1))

            curr_z_flat = curr_z.contiguous().view(B * C, self.d_agent)
            m_tau_flat = m_tau.contiguous().view(B * C, self.d_agent)
            curr_z_flat, h = self.agent_core.step(curr_z_flat, m_tau_flat, h)
            curr_z = curr_z_flat.view(B, C, self.d_agent)

        Z_sus_pred = torch.stack(Z_sus_pred, dim=1)  # (B, H, D)
        mean_dissonance = torch.stack(dissonances, dim=1).mean(dim=1)  # (B,)
        Z_ctx_flat = Z_ctx_agents.view(B, C_len, -1)  # (B, C_len, D)

        return Z_ctx_flat, Z_ctx_pred, Z_sus_pred, mean_dissonance

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = target_windows.shape[1] if target_windows is not None else 64
        Z_ctx, Z_ctx_pred, Z_sus_pred, dissonance = self.forward_latent_trajectory(
            context_windows, horizon=horizon
        )

        if target_windows is None:
            return Z_ctx.mean(dim=1), None, Z_sus_pred.mean(dim=1), Z_sus_pred, dissonance

        self.target_encoder.eval()
        with torch.no_grad():
            Z_tgt_agents = self.target_encoder(target_windows)
            Z_tgt = Z_tgt_agents.view(target_windows.shape[0], horizon, -1)

        return Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, dissonance

    def compute_objective(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        config=None,
        track_weight: float = 0.5,
        rollout_weight: float = 1.0,
        consensus_weight: float = 0.2,
        cov_weight: float = 0.5,
        var_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute context tracking + multi-agent rollout + consensus agreement + VICReg."""
        Z_ctx, Z_tgt, Z_ctx_pred, Z_sus_pred, dissonance = self.forward(ctx, tgt)

        # 1. 1-step tracking loss along context
        loss_track = torch.mean(torch.sum((Z_ctx[:, 1:] - Z_ctx_pred[:, :-1]) ** 2, dim=-1))

        # 2. Multi-agent rollout loss
        loss_rollout = torch.mean(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1))

        # 3. Inter-agent consensus agreement loss (regularizes message alignment)
        loss_consensus = torch.mean(dissonance)

        # 4. VICReg non-collapse
        z_flat = Z_ctx.contiguous().view(-1, self.latent_dim)
        std_c = torch.sqrt(torch.var(z_flat, dim=0, unbiased=False) + eps)
        var_c = torch.mean(F.relu(gamma - std_c))
        cov_c = von_neumann_operator_entropy_loss(z_flat, eps=eps)

        total_loss = (
            track_weight * loss_track
            + rollout_weight * loss_rollout
            + consensus_weight * loss_consensus
            + var_weight * var_c
            + cov_weight * cov_c
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_track": float(loss_track.item()),
            "loss_rollout": float(loss_rollout.item()),
            "loss_consensus": float(loss_consensus.item()),
            "mean_rollout_dist": float(torch.mean(torch.norm(Z_tgt - Z_sus_pred, dim=-1)).item()),
        }
        return total_loss, metrics

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Composite local sensor error + peer consensus dissonance scoring."""
        self.eval()
        horizon = observed_target_windows.shape[1]
        _, _, Z_sus_pred, dissonance = self.forward_latent_trajectory(context_windows, horizon=horizon)
        Z_tgt_agents = self.target_encoder(observed_target_windows)
        Z_tgt = Z_tgt_agents.view(observed_target_windows.shape[0], horizon, -1)

        # Local sensor rollout residuals: (B, H)
        step_residuals = torch.sqrt(torch.sum((Z_tgt - Z_sus_pred) ** 2, dim=-1) + 1e-8)
        mean_res = torch.mean(step_residuals, dim=-1)  # (B,)
        max_res = torch.max(step_residuals, dim=-1)[0]  # (B,)

        # Fusion: trajectory deviation + consensus dissonance
        total_score = mean_res + 0.5 * max_res + 0.5 * dissonance
        return total_score

    @torch.no_grad()
    def fit_mahalanobis_covariance(self, context_windows, target_windows, batch_size=512, reg=1e-3):
        def residual_fn(ctx_b, tgt_b):
            horizon = tgt_b.shape[1]
            _, _, Z_sus_pred, _ = self.forward_latent_trajectory(ctx_b, horizon=horizon)
            Z_tgt_agents = self.target_encoder(tgt_b)
            Z_tgt = Z_tgt_agents.view(tgt_b.shape[0], horizon, -1)
            return torch.mean(Z_tgt - Z_sus_pred, dim=1)

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


MultiAgentConsensusLatentJEPA = MultiAgentConsensusLatentJEPAModel
InterlatJEPA = MultiAgentConsensusLatentJEPAModel
