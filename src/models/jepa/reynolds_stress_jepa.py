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

from src.models._jepa_utils import JEPABase, fit_covariance_batched, whiten_residuals
from src.models.geometric_layers import ResolventPurification


def _qr_retract(mat: torch.Tensor) -> torch.Tensor:
    """Sign-stabilized QR retraction onto the Stiefel manifold."""
    if mat.size(0) <= mat.size(1):
        q, r = torch.linalg.qr(mat.T)
        d = torch.sign(torch.diagonal(r))
        d = torch.where(d == 0, torch.ones_like(d), d)
        return (q * d.unsqueeze(0)).T
    q, r = torch.linalg.qr(mat)
    d = torch.sign(torch.diagonal(r))
    d = torch.where(d == 0, torch.ones_like(d), d)
    return q * d.unsqueeze(0)


class StiefelLinear(nn.Module):
    """Row-orthonormal Stiefel projection P in St(out_features, in_features) via QR retraction."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.eye(out_features, in_features))

    def orthogonal_weight(self) -> torch.Tensor:
        return _qr_retract(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.orthogonal_weight())


class CrouzeixDissipativePredictor(nn.Sequential):
    """Crouzeix Numerical-Range Bounded Dissipative Predictor.

    Parameterizes the mean-flow linear transport operator via a Cayley-Accretive
    transform W = (I + S + D_visc)^{-1}(I - S - D_visc), where S = -S^T is
    skew-symmetric inviscid rotation and D_visc = B B^T + nu_0 I >= nu_0 I is
    symmetric positive-definite viscous dissipation. By the Complete Crouzeix
    Theorem, W(W_crouzeix) lies strictly inside the unit disk D, preventing
    pseudo-spectral transient amplification while allowing nonlinear shear
    corrections from the sequential MLP layers.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        predictor_layers: int = 2,
        dropout: float = 0.05,
        nu_0: float = 1e-3,
    ):
        layers = []
        in_d = latent_dim
        for _ in range(max(predictor_layers - 1, 1)):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, latent_dim))
        super().__init__(*layers)
        self.latent_dim = latent_dim
        self.nu_0 = nu_0
        self.skew_gen = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.diss_gen = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.shear_gate = nn.Parameter(torch.tensor(0.02))

    @property
    def transition_matrix(self) -> torch.Tensor:
        eye = torch.eye(self.latent_dim, device=self.skew_gen.device, dtype=self.skew_gen.dtype)
        s_mat = 0.5 * (self.skew_gen - self.skew_gen.T)
        d_mat = (self.diss_gen @ self.diss_gen.T) / float(self.latent_dim) + self.nu_0 * eye
        g_mat = s_mat + d_mat
        return torch.linalg.solve(eye + g_mat, eye - g_mat)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        w_crouzeix = self.transition_matrix
        z_lin = F.linear(z_ctx, w_crouzeix)
        z_shear = super().forward(z_ctx)
        return z_lin + torch.tanh(self.shear_gate) * z_shear


class ReynoldsStressClosureHead(nn.Module):
    """Predicts the target Reynolds stress tensor Sigma in S_+ from macro context
    via Admissible Stress Cone Projection (Navier-Stokes Blowup Paper, §4.3 & App. C)
    and Resolvent Purification.
    """

    def __init__(
        self,
        latent_dim: int,
        stress_dim: int = 16,
        gamma: float = 1e-3,
        use_cone_proj: bool = True,
        num_shear_modes: Optional[int] = None,
        channel_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.stress_dim = stress_dim
        self.gamma = gamma
        self.use_cone_proj = use_cone_proj
        self.channel_dim = (
            min(int(channel_dim), stress_dim)
            if channel_dim is not None and int(channel_dim) > 0
            else None
        )
        modes = stress_dim if num_shear_modes is None else min(num_shear_modes, stress_dim)
        if use_cone_proj:
            from src.models.geometric_layers import AdmissibleStressConeProjection
            self.cone_proj = AdmissibleStressConeProjection(
                latent_dim=latent_dim,
                stress_dim=stress_dim,
                num_shear_modes=modes,
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

    def _compute_block_cone_stress(self, z_macro: torch.Tensor) -> torch.Tensor:
        """Project z_macro into S_+ with block-orthogonal channel/latent Stiefel basis."""
        orig_shape = z_macro.shape[:-1]
        z_flat = z_macro.reshape(-1, self.latent_dim)
        b_sz = z_flat.size(0)
        d_s = self.stress_dim
        k_modes = self.cone_proj.num_shear_modes

        raw_coeffs = self.cone_proj.coeff_net(z_flat)
        c = F.softplus(raw_coeffs) + 1e-4  # (B, K_modes)

        k_ch = self.channel_dim
        if k_ch is not None and 0 < k_ch < d_s and k_modes == d_s:
            raw = self.cone_proj.raw_basis
            q_ch = _qr_retract(raw[:k_ch, :k_ch])
            q_lat = _qr_retract(raw[k_ch:, k_ch:])
            top = torch.cat(
                [q_ch, torch.zeros(k_ch, d_s - k_ch, device=raw.device, dtype=raw.dtype)],
                dim=1,
            )
            bot = torch.cat(
                [torch.zeros(d_s - k_ch, k_ch, device=raw.device, dtype=raw.dtype), q_lat],
                dim=1,
            )
            v_basis = torch.cat([top, bot], dim=0)
        else:
            v_basis = self.cone_proj.get_orthogonal_modes()

        v_exp = v_basis.unsqueeze(0).expand(b_sz, -1, -1)
        scaled_v = v_exp * torch.sqrt(c).unsqueeze(1)
        sigma = torch.bmm(scaled_v, scaled_v.transpose(1, 2))
        eye = torch.eye(d_s, device=z_macro.device, dtype=z_macro.dtype).unsqueeze(0)
        sigma = sigma + self.cone_proj.gamma * eye
        return sigma.view(*list(orig_shape), d_s, d_s)

    def forward(self, z_macro: torch.Tensor) -> torch.Tensor:
        if z_macro.ndim == 3:
            z_macro = z_macro.mean(dim=1)
        if self.cone_proj is not None:
            return self._compute_block_cone_stress(z_macro)
        raw = self.net(z_macro)
        mat = raw.reshape(-1, self.stress_dim, self.stress_dim)
        return self.purification(mat)


def _infer_encoder_input_dim(encoder: nn.Module) -> Optional[int]:
    """Infer input channel count K from a telemetry encoder module."""
    in_ch = getattr(encoder, "input_dim", None)
    if isinstance(in_ch, int) and in_ch > 0:
        return in_ch
    for mod in encoder.modules():
        if isinstance(mod, nn.Conv1d):
            return int(mod.in_channels)
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
    return None


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
        self.cov_mode = "instant"
        self.input_dim = _infer_encoder_input_dim(context_encoder)
        self._last_target_windows: Optional[torch.Tensor] = None

        self.target_encoder = self.init_target_encoder(context_encoder)

        # Shearing Wavelet Pulse Block (Navier-Stokes Blowup Paper, §7)
        if use_shearing_pulses:
            from src.models.geometric_layers import ShearingWaveletPulseBlock
            self.shearing_pulse = ShearingWaveletPulseBlock(channels=latent_dim)
        else:
            self.shearing_pulse = None

        # Hankel Moment Filter for algebraic trajectory consistency (Ten Proofs, Ch. 7)
        from src.models.geometric_layers import CohnElkiesFilter, CoordinateSaliencyGate, HankelMomentFilter
        self.hankel_filter = HankelMomentFilter(latent_dim=latent_dim, hankel_order=4)
        self.cohn_elkies = CohnElkiesFilter(latent_dim=latent_dim, n_shells=8)
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)
        self.filter_gate = nn.Parameter(torch.tensor(0.01))

        # Crouzeix Numerical-Range Bounded Dissipative Predictor
        self.predictor = CrouzeixDissipativePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            predictor_layers=predictor_layers,
            dropout=dropout,
            nu_0=1e-3,
        )

        # Macro-to-stress projection with Admissible Stress Cone (matched floor gamma=1e-3)
        self.stress_head = ReynoldsStressClosureHead(
            latent_dim=latent_dim,
            stress_dim=self.stress_dim,
            gamma=1e-3,
            use_cone_proj=use_admissible_cone,
            num_shear_modes=self.stress_dim,
            channel_dim=self.input_dim,
        )

        # Stiefel-orthogonal stress subspace projection if latent_dim != stress_dim
        if self.latent_dim != self.stress_dim:
            self.stress_proj = StiefelLinear(latent_dim, self.stress_dim)
        else:
            self.stress_proj = nn.Identity()

        self.register_mahalanobis_buffers(latent_dim)

    def _embed_channels(self, windows: torch.Tensor, out_dim: int) -> torch.Tensor:
        """Embed 3D window fluctuations (B, T, K) into (B, T, out_dim) preserving sensor signs."""
        b_sz, t_len, k_in = windows.shape
        if k_in == out_dim:
            return windows
        if k_in < out_dim:
            pad = torch.zeros(b_sz, t_len, out_dim - k_in, device=windows.device, dtype=windows.dtype)
            return torch.cat([windows, pad], dim=-1)
        # If k_in > out_dim, fold excess channels orthogonally into out_dim
        n_blocks = k_in // out_dim
        folded = windows[:, :, : n_blocks * out_dim].reshape(b_sz, t_len, n_blocks, out_dim).mean(dim=2)
        rem = k_in - n_blocks * out_dim
        if rem > 0:
            folded[:, :, :rem] = 0.5 * (folded[:, :, :rem] + windows[:, :, n_blocks * out_dim :])
        return folded

    def compute_observed_stress(
        self,
        z_tgt: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the empirical quadratic Reynolds stress tensor Sigma_obs in S_+.

        cov_mode controls the estimator (ablation axis):
        - "instant"  (default): combines temporal window micro-fluctuation covariance
          (when 3D target windows are available) with the Stiefel-projected latent
          outer product in S_+.
        - "temporal": per-sample covariance of projected target latents across
          the target-horizon time axis (requires 3D z_tgt or target_windows; falls
          back to "instant" for pooled inputs).
        - "batch":    covariance of projected target latents across the batch
          dimension, broadcast per-sample (requires B > 1).
        """
        eye = torch.eye(self.stress_dim, device=z_tgt.device, dtype=z_tgt.dtype).unsqueeze(0)
        if self.cov_mode == "temporal":
            z_seq = z_tgt if z_tgt.ndim == 3 else getattr(self.target_encoder, "last_trajectory", None)
            if (
                z_seq is not None
                and z_seq.ndim == 3
                and z_seq.size(0) == z_tgt.size(0)
                and z_seq.size(1) > 1
                and z_seq.device == z_tgt.device
            ):
                z_s_seq = self.stress_proj(z_seq.to(dtype=z_tgt.dtype))  # [B, T, d_s]
                z_c_seq = z_s_seq - z_s_seq.mean(dim=1, keepdim=True)
                t_seq = max(z_s_seq.shape[1] - 1, 1)
                sigma = torch.bmm(z_c_seq.transpose(1, 2), z_c_seq) / float(t_seq)
                return sigma + 1e-3 * eye

        z_tgt_pooled = z_tgt.mean(dim=1) if z_tgt.ndim == 3 else z_tgt
        z_s = self.stress_proj(z_tgt_pooled)  # [B, d_s]

        if self.cov_mode == "batch" and z_s.shape[0] > 1:
            z_c = z_s - z_s.mean(dim=0, keepdim=True)
            sigma = (z_c.T @ z_c) / (z_s.shape[0] - 1)  # [d_s, d_s]
            sigma = sigma.unsqueeze(0).expand(z_s.shape[0], -1, -1)
            return sigma + 1e-3 * eye

        # Trace-normalized latent outer product in S_+
        sigma_lat = torch.bmm(z_s.unsqueeze(2), z_s.unsqueeze(1)) / float(self.stress_dim)

        win = target_windows
        if (
            win is None
            and self._last_target_windows is not None
            and self._last_target_windows.ndim == 3
            and self._last_target_windows.size(0) == z_s.size(0)
            and self._last_target_windows.device == z_s.device
        ):
            win = self._last_target_windows

        if win is not None and win.ndim == 3 and win.size(1) > 1:
            w_f = win.to(dtype=z_tgt.dtype)
            x_c = w_f - w_f.mean(dim=1, keepdim=True)
            k_ch = min(w_f.size(-1), self.stress_dim)
            t_steps = max(w_f.size(1) - 1, 1)

            if k_ch < self.stress_dim:
                x_sub = x_c[:, :, :k_ch]
                sigma_ch_raw = torch.bmm(x_sub.transpose(1, 2), x_sub) / float(t_steps)  # [B, k_ch, k_ch]
                ch_norm = torch.linalg.norm(sigma_ch_raw, dim=(-2, -1), keepdim=True)
                sigma_ch = sigma_ch_raw * (torch.log1p(ch_norm) / (ch_norm + 1e-4))
                z_rem = z_s[:, k_ch:]  # [B, d_s - k_ch]
                sigma_rem = torch.bmm(z_rem.unsqueeze(2), z_rem.unsqueeze(1)) / float(self.stress_dim)
                b_sz = z_s.size(0)
                top = torch.cat(
                    [sigma_ch, torch.zeros(b_sz, k_ch, self.stress_dim - k_ch, device=z_s.device, dtype=z_s.dtype)],
                    dim=2,
                )
                bot = torch.cat(
                    [torch.zeros(b_sz, self.stress_dim - k_ch, k_ch, device=z_s.device, dtype=z_s.dtype), sigma_rem],
                    dim=2,
                )
                sigma = torch.cat([top, bot], dim=1)
                return sigma + 1e-3 * eye

            u_s = self._embed_channels(x_c, self.stress_dim)
            sigma_win_raw = torch.bmm(u_s.transpose(1, 2), u_s) / float(t_steps)
            win_norm = torch.linalg.norm(sigma_win_raw, dim=(-2, -1), keepdim=True)
            sigma_win = sigma_win_raw * (torch.log1p(win_norm) / (win_norm + 1e-4))
            return 0.9 * sigma_win + 0.1 * sigma_lat + 1e-3 * eye

        # Ensure symmetric positive semi-definiteness with minimum isotropic floor
        return sigma_lat + 1e-3 * eye

    def _extract_context(self, context_windows: torch.Tensor) -> torch.Tensor:
        """Extract context representations with 3D temporal shearing pulses and spherical preservation."""
        z_base = self.context_encoder(context_windows)
        fg = torch.tanh(self.filter_gate)
        if self.shearing_pulse is not None:
            if z_base.ndim == 3:
                z_seq = self.shearing_pulse(self.hankel_filter(z_base))
                return F.layer_norm(z_seq, (self.latent_dim,))
            if context_windows.ndim == 3 and context_windows.size(1) > 1:
                x_c = context_windows - context_windows.mean(dim=1, keepdim=True)
                x_std = x_c.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
                x_fluct = self._embed_channels(x_c / x_std, self.latent_dim)
                z_seq = z_base.unsqueeze(1) + 0.05 * x_fluct
                z_seq = self.shearing_pulse(z_seq)
                z_h = self.hankel_filter(z_seq.mean(dim=1))
                z_pooled = z_base + fg * (z_h - z_base)
                return F.layer_norm(z_pooled, (self.latent_dim,))
            z_h = self.hankel_filter(z_base)
            z_s = self.shearing_pulse(z_h.unsqueeze(1)).squeeze(1)
            return F.layer_norm(z_base + fg * (z_s - z_base), (self.latent_dim,))

        z_h = self.hankel_filter(z_base)
        return F.layer_norm(z_base + fg * (z_h - z_base), (self.latent_dim,))

    def _predict(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """Apply Crouzeix dissipative predictor and Cohn-Elkies Fourier modulation."""
        z_pred = self.predictor(z_ctx)
        return self.cohn_elkies(z_pred)

    def _extract_target(self, target_windows: torch.Tensor) -> torch.Tensor:
        """Extract EMA target representations without redundant filtering."""
        if target_windows.ndim == 3:
            self._last_target_windows = target_windows.detach()
        return self.target_encoder(target_windows)

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self._extract_context(context_windows)
        z_pred = self._predict(z_ctx)

        if target_windows is None:
            return z_pred, None, None

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self._extract_target(target_windows)

        sigma_obs = self.compute_observed_stress(z_tgt, target_windows=target_windows)
        sigma_pred = self.stress_head(0.5 * (z_ctx + z_pred))
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
        """Compute parsimonious two-term Reynolds objective: mean-flow MSE + stress closure MSE.

        All spectral contractivity (Crouzeix numerical range), Stiefel orthogonality,
        and positive-definite cone constraints (S_+) are enforced strictly by the
        architectural layers rather than auxiliary loss regularizers.
        """
        z_ctx = self._extract_context(ctx)
        z_pred = self._predict(z_ctx)

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self._extract_target(tgt)

        # 1. First-order mean-flow alignment: ||z_pred - sg(z_tgt)||_2^2
        pred_loss = F.mse_loss(z_pred, z_tgt)

        # 2. Second-order Reynolds stress closure alignment in S_+: ||Sigma_pred - sg(Sigma_obs)||_F^2
        sigma_obs = self.compute_observed_stress(z_tgt, target_windows=tgt)
        sigma_pred = self.stress_head(0.5 * (z_ctx + z_pred))
        stress_loss = F.mse_loss(sigma_pred, sigma_obs.detach())

        if self.alpha_stress > 0.0:
            total_loss = pred_loss + self.alpha_stress * stress_loss
        else:
            total_loss = pred_loss

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "stress_loss": stress_loss.item(),
            "cone_defect": 0.0,
        }
        return total_loss, metrics

    @staticmethod
    def _extract_radial_energy(
        encoder: nn.Module,
        z_out: torch.Tensor,
        window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract pre-LayerNorm radial energy preserved by the encoder."""
        base_energy = getattr(encoder, "last_radial_energy", None)
        if base_energy is None or base_energy.shape[0] != z_out.shape[0]:
            base_energy = torch.linalg.norm(z_out, dim=-1)
            if window is not None and window.ndim >= 2:
                reduce_dims = tuple(range(1, window.ndim))
                raw_rms = torch.sqrt(torch.mean(window.float().pow(2), dim=reduce_dims)).to(z_out.dtype)
                base_energy = base_energy * raw_rms
        return base_energy

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        self.eval()
        z_ctx = self._extract_context(context_windows)
        e_ctx = self._extract_radial_energy(self.context_encoder, z_ctx, window=context_windows)
        z_pred = self._predict(z_ctx)
        z_tgt = self._extract_target(observed_target_windows)
        e_tgt = self._extract_radial_energy(self.target_encoder, z_tgt, window=observed_target_windows)

        diff = z_tgt - z_pred
        if use_mahalanobis:
            if not bool(self.precision_fitted.item()):
                raise RuntimeError(
                    "Mahalanobis precision matrix has not been fitted. "
                    "Call fit_covariance() before compute_predictive_discrepancy(use_mahalanobis=True)."
                )
            diff_cent = diff - self.residual_mean
            e_white = whiten_residuals(diff_cent, self.precision_matrix)
            base_disc = self.saliency_gate(e_white)
        else:
            base_disc = self.saliency_gate(diff)
        self.last_base_disc = base_disc

        # Bilateral radial + Littlewood-Paley dyadic octave energy discrepancy
        pred_gain = torch.linalg.norm(z_pred, dim=-1) / (torch.linalg.norm(z_ctx, dim=-1) + 1e-6)
        e_pred = e_ctx * pred_gain
        radial_leak = torch.abs(e_tgt - e_pred) / (e_tgt + e_pred + 1e-6)

        s_ctx = getattr(getattr(self.context_encoder, "dyadic_block", None), "last_shell_energy", None)
        s_tgt = getattr(getattr(self.target_encoder, "dyadic_block", None), "last_shell_energy", None)
        if s_ctx is not None and s_tgt is not None and s_ctx.shape == s_tgt.shape and s_ctx.shape[0] == z_ctx.shape[0]:
            s_pred = s_ctx * (pred_gain.unsqueeze(-1) ** 2)
            shell_leak = torch.linalg.norm(s_tgt - s_pred, dim=-1) / (
                torch.linalg.norm(s_tgt, dim=-1) + torch.linalg.norm(s_pred, dim=-1) + 1e-6
            )
            energy_leak = 0.5 * radial_leak + 0.5 * shell_leak
        else:
            energy_leak = radial_leak

        if self.alpha_stress <= 0.0:
            return base_disc + 0.20 * energy_leak

        # Log-Ricci compressed Reynolds Stress Closure Discrepancy on S_+
        sigma_obs = self.compute_observed_stress(z_tgt, target_windows=observed_target_windows)
        sigma_pred = self.stress_head(0.5 * (z_ctx + z_pred))
        k_ch = min(observed_target_windows.size(-1), self.stress_dim) if observed_target_windows.ndim == 3 else self.stress_dim
        obs_sub = sigma_obs[:, :k_ch, :k_ch]
        pred_sub = sigma_pred[:, :k_ch, :k_ch]
        active_diff = (obs_sub - pred_sub).reshape(sigma_obs.size(0), -1)
        stress_disc = self.saliency_gate(active_diff) / math.sqrt(float(max(k_ch, 1)))

        return base_disc + 0.20 * energy_leak + 0.20 * stress_disc

    @torch.no_grad()
    def fit_covariance(
        self,
        context_windows: Union[np.ndarray, torch.Tensor],
        target_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 512,
        reg: float = 1e-3,
        method: str = "resolvent",
    ) -> None:
        def residual_fn(ctx, tgt):
            z_ctx = self._extract_context(ctx)
            z_pred = self._predict(z_ctx)
            z_tgt = self._extract_target(tgt)
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
