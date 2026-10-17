"""Von Neumann Operator Entropy & Spectral Rigidity JEPA Architecture.

Applies Von Neumann spectral entropy and a floored logarithmic Bregman barrier
to the latent linear transition operator K: \hat{z}_{tgt} = K z_{ctx}.
Prevents representation rank and energy collapse while detecting dynamical
anomalies via spectral dispersion and residual prediction errors.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched, whiten_residuals


def spectral_operator_entropy_loss(
    K: torch.Tensor,
    gamma_norm: float = 1.0,
    beta_norm: float = 1.0,
    eps_log: float = 1e-12,
    eps_safe: float = 1e-6,
    z_ctx: Optional[torch.Tensor] = None,
    proj_weight: float = 0.5,
    rank_ceiling: Optional[int] = None,
) -> torch.Tensor:
    """Anti-collapse Operator-Entropy regularizer (bare + data-projected).
    
    Penalizes both dimensional collapse (non-uniform spectrum) and energy collapse (K -> 0).
    When ``z_ctx`` is provided with batch size B > 1, combines bare ambient operator isotropy
    with data-projected operator entropy on the active Gram matrix ``K @ Sigma_ctx @ K^T``
    evaluated over the ``min(B - 1, D - 1)`` non-trivial LayerNorm subspace modes, preventing
    ``K`` from satisfying spectral isotropy exclusively on dead subspaces orthogonal to data.
    Falls back gracefully to bare operator entropy when ``z_ctx is None`` or ``B <= 1``.
    
    Args:
        K: Transition operator tensor of shape (D, D) or (B_k, D, D).
        gamma_norm: Target Frobenius energy scaling factor (default: 1.0 -> Tr(KK^T) = D).
        beta_norm: Weight of the logarithmic energy barrier.
        eps_log: Spectral probability smoothing constant for collapsing singular modes.
        eps_safe: Smooth barrier shift constant at zero energy.
        z_ctx: Optional context representation batch of shape (B, D) or (..., D) used to
            project K onto the active data covariance manifold.
        proj_weight: Blending weight in [0, 1] for the data-projected operator entropy term
            when ``z_ctx`` is provided (default: 0.5).
        rank_ceiling: Optional maximum active rank for data-projected covariance (defaults to
            ``min(B - 1, D - 1)`` to exclude the structural LayerNorm zero-sum nullspace).
    Returns:
        Scalar regularizer loss >= 0.
    """
    if K.ndim == 2:
        K = K.unsqueeze(0)
    B, D, _ = K.shape
    orig_dtype = K.dtype
    
    # 1. Upcast before Gram matrix multiplication to prevent float16/bfloat16 underflow/overflow
    K_work = K.float() if orig_dtype in (torch.float16, torch.bfloat16) else K
    KKt = torch.bmm(K_work, K_work.transpose(-1, -2))
    s_vals = torch.linalg.eigvalsh(KKt)  # (B, D), sorted ascending, >= 0
    # Straight-through non-negativity projection: clamps negative roundoff in forward pass
    # without zeroing gradients on collapsed modes where eigvalsh returns -1e-8 jitter
    s_vals = s_vals + (torch.clamp(s_vals, min=0.0) - s_vals).detach()
    
    # 2. Total operator energy Tr(KK^T) = ||K||_F^2
    energy = torch.sum(s_vals, dim=-1)  # (B,)
    target_energy = gamma_norm * float(D)
    
    # 3. Smoothed spectral distribution p_i = (s_i + eps_log) / sum_j (s_j + eps_log)
    # Ensures dL/ds_i < 0 for collapsing modes (s_i -> 0) instead of sign-reversing
    s_smooth = s_vals + eps_log
    p = s_smooth / torch.sum(s_smooth, dim=-1, keepdim=True)  # (B, D), sum_i p_i = 1
    
    # 4. Von Neumann Entropy H(p) = -sum_i p_i log(p_i)
    vn_entropy = -torch.sum(p * torch.log(p), dim=-1)  # (B,)
    
    max_entropy = math.log(float(D))
    entropy_deficiency = torch.clamp(max_entropy - vn_entropy, min=0.0)  # (B,)
    
    # 5. Smooth Shifted Itakura-Saito Energy Barrier:
    # b(r) = r - 1 - log(r + eps_safe) + log(1 + eps_safe) >= 0, smooth at r = 0
    energy_ratio = energy / target_energy
    energy_barrier = torch.clamp(
        energy_ratio - 1.0 - torch.log(energy_ratio + eps_safe) + math.log(1.0 + eps_safe),
        min=0.0,
    )
    
    bare_reg = entropy_deficiency + beta_norm * energy_barrier  # (B,)
    
    # 6. Optional Data-Projected Operator Entropy on K @ Sigma_ctx @ K^T
    if z_ctx is not None and proj_weight > 0.0:
        z_flat = z_ctx.reshape(-1, z_ctx.size(-1)) if z_ctx.ndim > 1 else z_ctx.unsqueeze(0)
        N_ctx = z_flat.size(0)
        if N_ctx > 1 and z_flat.size(-1) == D:
            z_work = z_flat.to(dtype=K_work.dtype, device=K_work.device)
            cov_ctx = compute_empirical_covariance(z_work)
            cov_sym = 0.5 * (cov_ctx + cov_ctx.transpose(-1, -2))
            tr_cov = torch.diagonal(cov_sym, dim1=-2, dim2=-1).sum()
            if tr_cov > eps_safe:
                default_ceiling = max(min(N_ctx - 1, max(D - 1, 1)), 1)
                eff_D = (
                    default_ceiling
                    if rank_ceiling is None
                    else min(max(int(rank_ceiling), 1), N_ctx - 1, D)
                )
                # Resolvent-tempered active-subspace metric P_ctx = Sigma_ctx (Sigma_ctx + tau I)^{-1}:
                # identically 0 on dead orthogonal modes (mu_i = 0) and uniform (~1) across active
                # modes, scaled so an isometric K on the active manifold has Tr(G_proj) = eff_D.
                mean_var = tr_cov / float(eff_D)
                evals_c, evecs_c = torch.linalg.eigh(cov_sym)
                evals_c = evals_c + (torch.clamp(evals_c, min=0.0) - evals_c).detach()
                sub_weights = evals_c / (evals_c + 0.01 * mean_var + eps_log)
                sub_scale = float(eff_D) / (torch.sum(sub_weights) + eps_safe)
                tilde_cov = (evecs_c * (sub_weights * sub_scale).unsqueeze(-2)) @ evecs_c.transpose(-1, -2)
                cov_batch = tilde_cov.unsqueeze(0).expand(B, -1, -1)
                G_proj = torch.bmm(torch.bmm(K_work, cov_batch), K_work.transpose(-1, -2))
                G_proj = 0.5 * (G_proj + G_proj.transpose(-1, -2))
                
                s_proj = torch.linalg.eigvalsh(G_proj)
                s_proj = s_proj + (torch.clamp(s_proj, min=0.0) - s_proj).detach()
                active_s = s_proj[..., D - eff_D :]
                
                proj_energy = torch.sum(active_s, dim=-1)
                target_proj_energy = gamma_norm * float(eff_D)
                
                s_proj_smooth = active_s + eps_log
                p_proj = s_proj_smooth / torch.sum(s_proj_smooth, dim=-1, keepdim=True)
                vn_proj = -torch.sum(p_proj * torch.log(p_proj), dim=-1)
                proj_deficiency = torch.clamp(math.log(float(eff_D)) - vn_proj, min=0.0)
                
                proj_ratio = proj_energy / target_proj_energy
                proj_barrier = torch.clamp(
                    proj_ratio - 1.0 - torch.log(proj_ratio + eps_safe) + math.log(1.0 + eps_safe),
                    min=0.0,
                )
                proj_reg = proj_deficiency + beta_norm * proj_barrier
                w = min(max(float(proj_weight), 0.0), 1.0)
                total_reg = (1.0 - w) * bare_reg + w * proj_reg
                return total_reg.mean().to(orig_dtype)
    
    return bare_reg.mean().to(orig_dtype)


def compute_empirical_covariance(z: torch.Tensor) -> torch.Tensor:
    """Compute empirical covariance matrix of latent embeddings z."""
    if z.ndim == 1:
        z = z.unsqueeze(0)
    elif z.ndim > 2:
        z = z.reshape(-1, z.size(-1))
    N = z.size(0)
    if N <= 1:
        return torch.zeros(z.size(-1), z.size(-1), device=z.device, dtype=z.dtype)
    z_c = z - z.mean(dim=0, keepdim=True)
    return (z_c.T @ z_c) / max(N - 1, 1)


def covariance_spectral_entropy(
    cov: torch.Tensor,
    eps_log: float = 1e-12,
    eps_safe: float = 1e-6,
    rank_ceiling: Optional[int] = None,
) -> torch.Tensor:
    """Compute scale-separated Von Neumann spectral entropy deficiency on representation covariance.
    
    Penalizes both dimensional collapse (non-uniform spectrum) and variance collapse (Tr(cov) -> 0).
    Supports an explicit rank_ceiling (e.g., D - 1 for LayerNorm embeddings that lie on the
    zero-mean hyperplane 1_D^T z = 0) so structural nullspace modes are excluded.
    """
    if cov.ndim == 2:
        cov = cov.unsqueeze(0)
    B, D, _ = cov.shape
    orig_dtype = cov.dtype

    # 1. Enforce exact symmetry and float32 precision for eigvalsh
    cov_work = cov.float() if orig_dtype in (torch.float16, torch.bfloat16) else cov
    cov_sym = 0.5 * (cov_work + cov_work.transpose(-1, -2))

    # 2. Compute eigenvalues of SPSD covariance matrix (sorted ascending)
    evals = torch.linalg.eigvalsh(cov_sym)
    evals = evals + (torch.clamp(evals, min=0.0) - evals).detach()

    eff_D = D if rank_ceiling is None else min(max(int(rank_ceiling), 1), D)
    active_evals = evals[..., D - eff_D :]

    # 3. Total variance Tr(cov) and maximum entropy ceiling log(eff_D)
    total_var = torch.sum(active_evals, dim=-1)
    max_entropy = math.log(float(eff_D))

    # 4. Smoothed spectral distribution p_i = (lambda_i + eps_log) / sum_j (lambda_j + eps_log)
    evals_smooth = active_evals + eps_log
    p = evals_smooth / torch.sum(evals_smooth, dim=-1, keepdim=True)

    # 5. Von Neumann Entropy
    vn_entropy = -torch.sum(p * torch.log(p), dim=-1)
    entropy_deficiency = torch.clamp(max_entropy - vn_entropy, min=0.0)

    # 6. Smooth Shifted Itakura-Saito Variance Barrier
    var_ratio = total_var / float(D)
    var_barrier = torch.clamp(
        var_ratio - 1.0 - torch.log(var_ratio + eps_safe) + math.log(1.0 + eps_safe),
        min=0.0,
    )

    total_reg = entropy_deficiency + var_barrier
    return total_reg.mean().to(orig_dtype)


def von_neumann_entropy(
    z: torch.Tensor,
    eps_log: float = 1e-12,
    eps_safe: float = 1e-6,
    rank_ceiling: Optional[int] = None,
) -> torch.Tensor:
    """Compute scale-separated Von Neumann operator entropy deficiency on covariance.
    
    Strictly penalizes both rank deficiency and variance collapse (z -> 0).
    """
    if z.ndim == 3:
        z = z.reshape(-1, z.size(-1))
    N = z.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    cov = compute_empirical_covariance(z)
    return covariance_spectral_entropy(
        cov, eps_log=eps_log, eps_safe=eps_safe, rank_ceiling=rank_ceiling
    )


class KoopmanLinearTransition(nn.Module):
    """Stateless hookable module applying the explicit Koopman transition z @ K^T."""

    def forward(self, z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        return z @ K.T


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
        alpha_rep_entropy: float = 0.0,
        beta_lyapunov: float = 0.02,
        spectral_gap_min: float = 0.1,
        dropout: float = 0.05,
        use_radial_heat: bool = False,
        use_cohn_elkies: bool = False,
        use_koopman_matrix: bool = True,
        use_cayley_operator: bool = False,
        proj_weight: float = 0.2,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.alpha_entropy = alpha_entropy
        self.alpha_rep_entropy = alpha_rep_entropy
        self.beta_lyapunov = beta_lyapunov
        self.spectral_gap_min = spectral_gap_min
        self.use_radial_heat = use_radial_heat
        self.use_cohn_elkies = use_cohn_elkies
        self.use_koopman_matrix = use_koopman_matrix
        self.use_cayley_operator = use_cayley_operator
        self.proj_weight = proj_weight

        self.target_encoder = self.init_target_encoder(context_encoder)

        if use_cayley_operator:
            # Cayley transform orthogonal-dissipative gate: W = (I+S+D)^{-1}(I-S-D)
            # Enforces ||W||_2 <= 1.0 and rho(W) <= 1.0 strictly by algebraic construction
            from src.models.geometric_layers import CayleyOrthogonalGate
            self.cayley_gate = CayleyOrthogonalGate(dim=latent_dim, bias=False)
            self.K = None
            self.koopman_op = None
            self.predictor = None
        elif use_koopman_matrix:
            # Explicit Koopman transition matrix K in R^{D x D} (1,024 parameters for D=32)
            # Initialized near identity with small random perturbations
            self.K = nn.Parameter(torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim))
            self.koopman_op = KoopmanLinearTransition()
            self.cayley_gate = None
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
            self.koopman_op = None
            self.cayley_gate = None

        # ResolventPurification instantiation removed (unused in forward/loss)
        self.resolvent = None

        # Radial heat exterior boundary damping (excised by default; inert math layer)
        if use_radial_heat:
            from src.models.geometric_layers import RadialHeatExteriorBoundary
            self.radial_heat = RadialHeatExteriorBoundary(latent_dim=latent_dim)
        else:
            self.radial_heat = None

        # Cohn-Elkies Fourier modulation layer (excised by default; inert math layer)
        if use_cohn_elkies:
            from src.models.geometric_layers import CohnElkiesFilter
            self.cohn_elkies = CohnElkiesFilter(latent_dim=latent_dim, n_shells=8)
        else:
            self.cohn_elkies = None

        from src.models.geometric_layers import CoordinateSaliencyGate
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)

        self.register_mahalanobis_buffers(latent_dim)

    def _predict(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the predictive transition operator."""
        if self.use_cayley_operator and self.cayley_gate is not None:
            return self.cayley_gate(z)
        if self.use_koopman_matrix and self.K is not None:
            if self.koopman_op is not None:
                return self.koopman_op(z, self.K)
            return z @ self.K.T
        return self.predictor(z)

    @property
    def effective_transition_matrix(self) -> Optional[torch.Tensor]:
        """Return the active linear transition operator matrix if applicable."""
        if self.use_cayley_operator and self.cayley_gate is not None:
            return self.cayley_gate.transition_matrix
        if self.use_koopman_matrix and self.K is not None:
            return self.K
        return None

    def _apply_filters(self, z: torch.Tensor) -> torch.Tensor:
        if self.radial_heat is not None:
            z = self.radial_heat(z)
        if self.cohn_elkies is not None:
            z = self.cohn_elkies(z)
        return z

    def forward(
        self,
        context_windows: torch.Tensor,
        target_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ctx = self._apply_filters(self.context_encoder(context_windows))
        z_pred = self._predict(z_ctx)

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
        alpha_rep_entropy: Optional[float] = None,
        beta_lyapunov: Optional[float] = None,
        proj_weight: Optional[float] = None,
        gap_weight: float = 0.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        z_ctx = self._apply_filters(self.context_encoder(ctx))
        z_pred = self._predict(z_ctx)

        self.target_encoder.eval()
        with torch.no_grad():
            z_tgt = self._apply_filters(self.target_encoder(tgt))

        # 1. Pure predictive MSE loss
        pred_loss = F.mse_loss(z_pred, z_tgt)

        # 2. Dual Spectral Entropy Coupling (Operator K + Representation Covariance Sigma_Z)
        effective_alpha = (
            alpha_entropy
            if alpha_entropy is not None
            else getattr(config, "alpha_entropy", self.alpha_entropy)
        )
        effective_alpha_rep = (
            alpha_rep_entropy
            if alpha_rep_entropy is not None
            else getattr(config, "alpha_rep_entropy", self.alpha_rep_entropy)
        )
        effective_beta_lyap = (
            beta_lyapunov
            if beta_lyapunov is not None
            else getattr(config, "beta_lyapunov", self.beta_lyapunov)
        )
        effective_proj_weight = (
            proj_weight
            if proj_weight is not None
            else getattr(config, "proj_weight", self.proj_weight)
        )

        if effective_alpha > 0:
            if self.use_cayley_operator and self.cayley_gate is not None:
                # Cayley operator W is contractive (||W||_2 <= 1.0, rho(W) <= 1.0) by construction;
                # disable the Frobenius energy barrier (beta_norm=0.0) so dissipative damping is not penalized.
                W = self.cayley_gate.transition_matrix
                reg_K = spectral_operator_entropy_loss(
                    W,
                    gamma_norm=1.0,
                    beta_norm=0.0,
                    z_ctx=z_ctx.detach(),
                    proj_weight=effective_proj_weight,
                )
            elif self.use_koopman_matrix and self.K is not None:
                reg_K = spectral_operator_entropy_loss(
                    self.K,
                    gamma_norm=1.0,
                    beta_norm=1.0,
                    z_ctx=z_ctx.detach(),
                    proj_weight=effective_proj_weight,
                )
            else:
                reg_K = torch.tensor(0.0, device=pred_loss.device, dtype=pred_loss.dtype)
        else:
            reg_K = torch.tensor(0.0, device=pred_loss.device, dtype=pred_loss.dtype)

        num_samples = z_ctx.numel() // z_ctx.size(-1)
        cov_ctx = None
        cov_tgt = None
        if effective_alpha_rep > 0 and num_samples > 1:
            cov_ctx = compute_empirical_covariance(z_ctx)
            cov_tgt = compute_empirical_covariance(z_tgt)
            eff_rank = max(self.latent_dim - 1, 1)
            reg_Z = 0.5 * (
                covariance_spectral_entropy(cov_ctx, rank_ceiling=eff_rank)
                + covariance_spectral_entropy(cov_tgt, rank_ceiling=eff_rank)
            )
        else:
            reg_Z = torch.tensor(0.0, device=pred_loss.device, dtype=pred_loss.dtype)

        # 3. Scale-Invariant Discrete Lyapunov Invariance Defect on the transition operator:
        # Evaluates operator covariance alignment on detached z_ctx with horizon-trace normalization
        # so difference in window length (C=256 vs S=64) does not collapse context_encoder onto top PC.
        if effective_beta_lyap > 0 and num_samples > 1:
            if cov_ctx is None:
                cov_ctx = compute_empirical_covariance(z_ctx.detach())
            if cov_tgt is None:
                cov_tgt = compute_empirical_covariance(z_tgt)
            z_pred_lyap = self._predict(z_ctx.detach())
            cov_pred = compute_empirical_covariance(z_pred_lyap)
            ctx_scale = (
                torch.diagonal(cov_ctx.detach(), dim1=-2, dim2=-1).sum(dim=-1)
                / float(self.latent_dim)
                + 1e-6
            )
            tgt_scale = (
                torch.diagonal(cov_tgt, dim1=-2, dim2=-1).sum(dim=-1)
                / float(self.latent_dim)
                + 1e-6
            )
            lyapunov_defect = (
                torch.norm(cov_pred / ctx_scale - cov_tgt / tgt_scale, p="fro") ** 2
                / float(self.latent_dim)
            )
        else:
            lyapunov_defect = torch.tensor(0.0, device=pred_loss.device, dtype=pred_loss.dtype)

        total_loss = (
            pred_loss
            + effective_alpha * reg_K
            + effective_alpha_rep * reg_Z
            + effective_beta_lyap * lyapunov_defect
        )

        # Compute spectral gap and variance hinge with gradients only when their weights are active
        z_p = z_pred.reshape(-1, z_pred.size(-1)) if z_pred.ndim > 2 else z_pred
        with torch.set_grad_enabled(gap_weight > 0 and z_p.requires_grad):
            z_cent = z_p - z_p.mean(dim=0, keepdim=True)
            cov = (z_cent.T @ z_cent) / max(z_p.size(0) - 1, 1)
            cov_sym = 0.5 * (cov + cov.T)
            cov_work = cov_sym.float() if cov_sym.dtype in (torch.float16, torch.bfloat16) else cov_sym
            evals = torch.linalg.eigvalsh(cov_work)
            gap = evals[-1] - evals[0]
            gap_loss = F.relu(self.spectral_gap_min - gap).to(pred_loss.dtype)

        with torch.set_grad_enabled(var_weight > 0 and z_p.requires_grad):
            std_pred = torch.sqrt(z_p.var(dim=0, unbiased=False) + eps)
            std_loss = torch.mean(F.relu(gamma - std_pred))

        if var_weight > 0:
            total_loss = total_loss + var_weight * std_loss
        if gap_weight > 0:
            total_loss = total_loss + gap_weight * gap_loss

        metrics = {
            "loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "reg_K": reg_K.item(),
            "reg_Z": reg_Z.item(),
            "lyapunov_defect": lyapunov_defect.item(),
            "spectral_reg": (
                effective_alpha * reg_K
                + effective_alpha_rep * reg_Z
                + effective_beta_lyap * lyapunov_defect
            ).item(),
            "vn_loss": reg_Z.item(),
            "gap_loss": gap_loss.item(),
            "std_loss": std_loss.item(),
        }
        return total_loss, metrics

    def _apply_radial_heat(self, z: torch.Tensor) -> torch.Tensor:
        """Backward compatible alias."""
        return self._apply_filters(z)

    @staticmethod
    def _extract_radial_energy(
        encoder: nn.Module,
        z_raw: torch.Tensor,
        z_filtered: torch.Tensor,
        window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract pre-LayerNorm radial energy preserved by the encoder."""
        base_energy = getattr(encoder, "last_radial_energy", None)
        if base_energy is None or base_energy.shape[0] != z_raw.shape[0]:
            base_energy = torch.linalg.norm(z_raw, dim=-1)
            if window is not None and window.ndim >= 2 and any(
                isinstance(m, nn.LayerNorm) for m in encoder.modules()
            ):
                reduce_dims = tuple(range(1, window.ndim))
                raw_rms = torch.sqrt(torch.mean(window.pow(2), dim=reduce_dims))
                base_energy = base_energy * raw_rms
        filter_scale = torch.linalg.norm(z_filtered, dim=-1) / (
            torch.linalg.norm(z_raw, dim=-1) + 1e-6
        )
        return base_energy * filter_scale

    @torch.no_grad()
    def compute_predictive_discrepancy(
        self,
        context_windows: torch.Tensor,
        observed_target_windows: torch.Tensor,
        use_mahalanobis: bool = False,
    ) -> torch.Tensor:
        self.eval()
        z_ctx_raw = self.context_encoder(context_windows)
        z_ctx = self._apply_filters(z_ctx_raw)
        e_ctx = self._extract_radial_energy(
            self.context_encoder, z_ctx_raw, z_ctx, window=context_windows
        )

        z_pred = self._predict(z_ctx)

        z_tgt_raw = self.target_encoder(observed_target_windows)
        z_tgt = self._apply_filters(z_tgt_raw)
        e_tgt = self._extract_radial_energy(
            self.target_encoder, z_tgt_raw, z_tgt, window=observed_target_windows
        )

        diff = z_tgt - z_pred
        if use_mahalanobis:
            if not bool(self.precision_fitted.item()):
                raise RuntimeError(
                    "Mahalanobis precision matrix has not been fitted. "
                    "Call fit_covariance() before compute_predictive_discrepancy(use_mahalanobis=True)."
                )
            diff_cent = diff - self.residual_mean
            e_white = whiten_residuals(diff_cent, self.precision_matrix)
            disc = self.saliency_gate(e_white)
        else:
            disc = self.saliency_gate(diff)

        # Bilateral relative energy discrepancy using pre-LayerNorm radial energy
        pred_gain = torch.linalg.norm(z_pred, dim=-1) / (torch.linalg.norm(z_ctx, dim=-1) + 1e-6)
        e_pred = e_ctx * pred_gain
        energy_leak = torch.abs(e_tgt - e_pred) / (e_tgt + e_pred + 1e-6)
        return disc + 0.2 * energy_leak

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
            z_ctx = self._apply_filters(self.context_encoder(ctx))
            z_pred = self._predict(z_ctx)
            z_tgt = self._apply_filters(self.target_encoder(tgt))
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
