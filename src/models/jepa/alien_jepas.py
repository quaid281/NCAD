"""Complete consolidated Alien & Spherical Fluid JEPA models."""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._jepa_utils import JEPABase, fit_covariance_batched, whiten_residuals
from src.models.geometric_layers import CoordinateSaliencyGate
from src.models.jepa.ts_jepa import LatentPredictor, jepa_vicreg_loss, log_det_whitening_loss

def _disable_dropout(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.p = 0.0

def _init_jepa_encoders(model: JEPABase, context_encoder: nn.Module, latent_dim: int, ema_decay: float) -> None:
    _disable_dropout(context_encoder)
    model.context_encoder = context_encoder
    model.latent_dim, model.ema_decay = latent_dim, ema_decay
    model.target_encoder = model.init_target_encoder(context_encoder)
    _disable_dropout(model.target_encoder)

class MultiDepthTapHead(nn.Module):
    """Intermediate Layer-Wise Functional Hierarchy Probe (Alien Atoms [4, 207])."""

    def __init__(self, filters: int, latent_dim: int):
        super().__init__()
        in_dim = filters * 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * latent_dim, bias=False), nn.GELU(), nn.LayerNorm(2 * latent_dim, elementwise_affine=False),
            nn.Linear(2 * latent_dim, latent_dim, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(0.35))
        self.register_buffer("running_centroid", torch.zeros(latent_dim))
        self.register_buffer("has_centroid", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _tap_stats(x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(dim=-1, keepdim=True)
        return torch.cat([xc[:, :, -1], torch.sqrt(xc.pow(2).mean(dim=-1) + 1e-6)], dim=1)

    def forward(self, z_deep: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        d = (z_deep.shape[-1],)
        tap_vec = torch.cat([self._tap_stats(t) for t in taps], dim=1)
        g = 0.5 * torch.sigmoid(self.gate)
        fused = (1.0 - g) * z_deep + g * F.layer_norm(self.net(tap_vec), d)
        if self.training:
            with torch.no_grad():
                c = fused.detach().mean(dim=0)
                if not bool(self.has_centroid.item()):
                    self.running_centroid.copy_(c)
                    self.has_centroid.fill_(True)
                else:
                    self.running_centroid.lerp_(c, 0.05)
            centered = fused - self.running_centroid.unsqueeze(0)
        elif bool(self.has_centroid.item()):
            centered = fused - self.running_centroid.unsqueeze(0)
        elif fused.shape[0] > 1:
            centered = fused - fused.mean(dim=0, keepdim=True)
        else:
            centered = fused
        return F.layer_norm(centered, d)

def _encode_with_multidepth_taps(encoder: nn.Module, x: torch.Tensor, tap_head: Optional[MultiDepthTapHead]) -> torch.Tensor:
    if tap_head is None or not hasattr(encoder, "blocks") or len(getattr(encoder, "blocks", [])) == 0:
        return encoder(x)
    captured: dict[str, torch.Tensor] = {}
    blocks = encoder.blocks
    shallow_mod = getattr(encoder, "dyadic_block", None) or getattr(encoder, "input_projection", blocks[0])
    handles = [
        shallow_mod.register_forward_hook(lambda _m, _i, out: captured.update({"t0": out}) if isinstance(out, torch.Tensor) else None),
        blocks[max(0, (len(blocks) - 1) // 3)].register_forward_hook(lambda _m, _i, out: captured.update({"t1": out}) if isinstance(out, torch.Tensor) else None),
        blocks[-1].register_forward_hook(lambda _m, _i, out: captured.update({"t2": out}) if isinstance(out, torch.Tensor) else None),
    ]
    try:
        z_deep = encoder(x)
    finally:
        for h in handles:
            h.remove()
    return tap_head(z_deep, [captured["t0"], captured["t1"], captured["t2"]]) if len(captured) == 3 else z_deep

def _init_tap_heads(model: nn.Module, context_encoder: nn.Module, latent_dim: int) -> None:
    filters = int(getattr(context_encoder, "filters", latent_dim))
    model.context_tap_head = MultiDepthTapHead(filters=filters, latent_dim=latent_dim)
    model.target_tap_head = copy.deepcopy(model.context_tap_head)
    for p in model.target_tap_head.parameters():
        p.requires_grad = False

def _update_tap_ema(model: nn.Module, momentum: Optional[float] = None) -> None:
    m = model.ema_decay if momentum is None else momentum
    for p_t, p_c in zip(model.target_tap_head.parameters(), model.context_tap_head.parameters()):
        p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)
    model.target_tap_head.running_centroid.copy_(model.context_tap_head.running_centroid)
    model.target_tap_head.has_centroid.copy_(model.context_tap_head.has_centroid)

def _verify_3horizons(model: nn.Module, tgt_win: torch.Tensor) -> torch.Tensor:
    S_len = tgt_win.shape[1]
    h0, h1 = max(S_len // 4, 2), max(S_len // 2, 4)
    model.target_encoder.eval()
    model.target_tap_head.eval()
    with torch.no_grad():
        t0 = _encode_with_multidepth_taps(model.target_encoder, tgt_win[:, :h0, :], model.target_tap_head)
        t1 = _encode_with_multidepth_taps(model.target_encoder, tgt_win[:, :h1, :], model.target_tap_head)
        t2 = _encode_with_multidepth_taps(model.target_encoder, tgt_win, model.target_tap_head)
    return torch.stack([t0, t1, t2], dim=1)

def _multihorizon_vicreg_loss(z_drafts: torch.Tensor, z_targets: torch.Tensor, z_ctx: Optional[torch.Tensor] = None, weights: Tuple[float, ...] = (0.20, 0.25, 0.55)) -> torch.Tensor:
    total = torch.tensor(0.0, device=z_drafts.device)
    for h, w in enumerate(weights):
        ctx_ref = z_ctx if h == len(weights) - 1 else None
        lh = jepa_vicreg_loss(z_drafts[:, h, :], z_targets[:, h, :], ctx_ref, sim_weight=1.0, var_weight=1.0, cov_weight=0.5, gamma=0.75, use_log_det=False)
        total = total + w * (lh + log_det_whitening_loss(z_drafts[:, h, :], beta=0.05))
    return total

def _register_alien_bank(module: nn.Module, latent_dim: int, coreset_size: int = 64, top_k: int = 5, null_dim: int = 8) -> None:
    module.coreset_size, module.top_k_prototypes, module.null_dim = coreset_size, top_k, null_dim
    for name, t in [("coreset_bank", torch.zeros(coreset_size, 2 * latent_dim)),
                    ("joint_center", torch.zeros(2 * latent_dim)),
                    ("null_projector", torch.eye(latent_dim)),
                    ("unavail_med", torch.tensor(1.0)),
                    ("unavail_mad", torch.tensor(1.0)),
                    ("null_ratio_med", torch.tensor(1.0)),
                    ("bank_fitted", torch.tensor(False))]:
        module.register_buffer(name, t)


def _init_multihorizon_alien_gates_and_buffers(module: nn.Module, latent_dim: int, num_horizons: int, coreset_size: int, top_k: int, null_dim: int) -> None:
    module.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)
    module.joint_saliency_gate = CoordinateSaliencyGate(dim=num_horizons * latent_dim, tau=0.5, alpha=0.5)
    module.register_mahalanobis_buffers(num_horizons * latent_dim)
    _register_alien_bank(module, latent_dim, coreset_size, top_k, null_dim)

def _fit_coreset_and_null_projector(
    J: torch.Tensor, R: torch.Tensor, coreset_size: int, top_k_prototypes: int, null_dim: int,
    joint_center: torch.Tensor, null_projector: torch.Tensor, null_ratio_med: torch.Tensor,
    coreset_bank: torch.Tensor, unavail_med: torch.Tensor, unavail_mad: torch.Tensor, bank_fitted: torch.Tensor,
) -> None:
    center = J.mean(dim=0)
    joint_center.copy_(center)
    J_norm = F.normalize(J - center, p=2, dim=-1)
    R_c = R - R.mean(dim=0, keepdim=True)
    cov_r = (R_c.T @ R_c) / max(len(R) - 1, 1)
    _, eigvecs = torch.linalg.eigh(cov_r)
    null_proj = eigvecs[:, :null_dim] @ eigvecs[:, :null_dim].T
    null_projector.copy_(null_proj)
    null_frac = (R @ null_proj).norm(dim=-1) / (R.norm(dim=-1) + 1e-6)
    null_ratio_med.copy_(torch.median(null_frac).clamp(min=1e-4))

    M = min(coreset_size, len(J_norm))
    selected, min_dists = [0], 1.0 - (J_norm @ J_norm[0])
    for _ in range(1, M):
        idx = int(torch.argmax(min_dists).item())
        selected.append(idx)
        min_dists = torch.minimum(min_dists, 1.0 - (J_norm @ J_norm[idx]))
    coreset = J_norm[selected] if M >= coreset_size else torch.cat([J_norm[selected], J_norm[:1].repeat(coreset_size - M, 1)], dim=0)
    coreset_bank.copy_(coreset)

    k = min(top_k_prototypes, coreset.size(0))
    null_unavail = 1.0 - (J_norm @ coreset.T).topk(k, dim=1).values[:, k // 2]
    med = torch.median(null_unavail)
    unavail_med.copy_(med)
    unavail_mad.copy_(torch.median(torch.abs(null_unavail - med)).clamp(min=1e-4))
    bank_fitted.fill_(True)

def _collect_alien_joints_resids(model: nn.Module, train_windows: np.ndarray, context_len: int, batch_size: int, is_speculative: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    joints, resids = [], []
    for i in range(0, len(train_windows), batch_size):
        b = torch.from_numpy(train_windows[i : i + batch_size]).float().to(device)
        zc, zt, zp = model.forward(b[:, :context_len], b[:, context_len:])[:3]
        zt_f = zt[:, 2, :] if is_speculative else zt
        zp_f = zp[:, 2, :] if is_speculative else zp
        joints.append(torch.cat([zc, zt_f], dim=-1))
        resids.append(zt_f - zp_f)
    return torch.cat(joints, dim=0), torch.cat(resids, dim=0)

def _fit_alien_bank_default(model: nn.Module, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
    model.eval()
    J, R = _collect_alien_joints_resids(model, train_windows, context_len, batch_size, is_speculative=False)
    _fit_coreset_and_null_projector(
        J, R, model.coreset_size, model.top_k_prototypes, model.null_dim,
        model.joint_center, model.null_projector, model.null_ratio_med,
        model.coreset_bank, model.unavail_med, model.unavail_mad, model.bank_fitted,
    )

def _fit_multihorizon_alien_pipeline(model: nn.Module, train_windows, context_len: int = 256, batch_size: int = 256, residual_fn=None, is_speculative: bool = False) -> None:
    model.eval()
    ctx_all, tgt_all = train_windows[:, :context_len], train_windows[:, context_len:]
    fit_covariance_batched(
        model, ctx_all, tgt_all, residual_fn=residual_fn,
        dim=model.num_horizons * model.latent_dim, batch_size=batch_size, reg=1e-3,
        precision_buffer=model.precision_matrix, residual_mean_buffer=model.residual_mean, fitted_buffer=model.precision_fitted,
    )
    J, R = _collect_alien_joints_resids(model, train_windows, context_len, batch_size, is_speculative=is_speculative)
    _fit_coreset_and_null_projector(
        J, R, model.coreset_size, model.top_k_prototypes, model.null_dim,
        model.joint_center, model.null_projector, model.null_ratio_med,
        model.coreset_bank, model.unavail_med, model.unavail_mad, model.bank_fitted,
    )

def _score_multihorizon_envelope(model: nn.Module, diffs: torch.Tensor, shear_gain: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, D = diffs.shape
    if bool(model.precision_fitted.item()):
        e_white = whiten_residuals(diffs.reshape(B, H * D) - model.residual_mean, model.precision_matrix).reshape(B, H, D)
        e_eff = e_white * shear_gain if shear_gain is not None else e_white
        s_joint = model.joint_saliency_gate(e_eff.reshape(B, H * D))
        d_h = model.saliency_gate(e_eff.reshape(B * H, D)).reshape(B, H)
    else:
        e_eff = diffs * shear_gain if shear_gain is not None else diffs
        d_h = model.saliency_gate(e_eff.reshape(B * H, D)).reshape(B, H)
        s_joint = d_h.mean(dim=-1)
    s_env = torch.sqrt(0.5 * (s_joint.pow(2) + d_h.pow(2).amax(dim=-1)))
    return s_env, e_eff - e_eff.mean(dim=1, keepdim=True)

def _alien_bank_gain(
    z_ctx: torch.Tensor, z_tgt: torch.Tensor, diff: torch.Tensor, joint_center: torch.Tensor,
    coreset_bank: torch.Tensor, top_k: int, unavail_med: torch.Tensor, unavail_mad: torch.Tensor,
    null_projector: torch.Tensor, null_ratio_med: torch.Tensor, w_u: float, w_n: float,
) -> torch.Tensor:
    j_state = torch.cat([z_ctx, z_tgt], dim=-1) - joint_center.unsqueeze(0)
    k = min(top_k, coreset_bank.size(0))
    unavail = 1.0 - (F.normalize(j_state, p=2, dim=-1) @ coreset_bank.T).topk(k, dim=1).values[:, k // 2]
    z_u = (unavail - unavail_med) / (1.4826 * unavail_mad)
    null_frac = (diff @ null_projector).norm(dim=-1) / (diff.norm(dim=-1) + 1e-6)
    z_n = (null_frac - null_ratio_med) / null_ratio_med
    return 1.0 + w_u * torch.tanh(0.5 * z_u) + w_n * F.relu(torch.tanh(z_n))

def _apply_bank_gain(model: nn.Module, z_ctx, z_tgt, diff, base_score, w_u: float = 0.20, w_n: float = 0.15):
    if not bool(model.bank_fitted.item()):
        return base_score
    gain = _alien_bank_gain(
        z_ctx, z_tgt, diff, model.joint_center, model.coreset_bank,
        model.top_k_prototypes, model.unavail_med, model.unavail_mad,
        model.null_projector, model.null_ratio_med, w_u, w_n,
    )
    return base_score * gain

def _encode_causal_subblocks(context_windows: torch.Tensor, num_subblocks: int, latent_dim: int, context_encoder: nn.Module, target_encoder: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C_len, K = context_windows.shape
    sub_len = C_len // num_subblocks
    traj = getattr(context_encoder, "last_trajectory", None)
    if traj is not None and traj.shape[0] == B and traj.shape[1] >= num_subblocks:
        L = traj.shape[1] // num_subblocks
        z_sub_ctx = F.layer_norm(traj[:, : num_subblocks * L, :].reshape(B, num_subblocks, L, latent_dim).mean(dim=2), (latent_dim,))
        return z_sub_ctx, z_sub_ctx.detach()
    flat_sub = context_windows[:, : num_subblocks * sub_len, :].reshape(B * num_subblocks, sub_len, K)
    z_sub_ctx = context_encoder(flat_sub).reshape(B, num_subblocks, latent_dim)
    target_encoder.eval()
    with torch.no_grad():
        z_sub_tgt = target_encoder(flat_sub).reshape(B, num_subblocks, latent_dim)
    return z_sub_ctx, z_sub_tgt

def _rollout_solenoidal_vorticity(
    predictor: Optional[nn.Module], z_sub_ctx: torch.Tensor, z_sub_tgt: torch.Tensor,
    decay: float = 0.85, lr: float = 0.25, cs: float = 0.16, nu_0: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, M, D = z_sub_ctx.shape
    scale_d = math.sqrt(float(D))
    omega = torch.zeros(B, D, D, device=z_sub_ctx.device, dtype=z_sub_ctx.dtype)
    strains = []
    for m in range(M - 1):
        zm, zn_tgt = z_sub_ctx[:, m, :], z_sub_tgt[:, m + 1, :]
        um = zm / (zm.norm(dim=-1, keepdim=True) + 1e-6)
        base_pred = F.layer_norm(predictor(zm), (D,)) if predictor is not None else zm
        rm = zn_tgt - (base_pred + scale_d * torch.bmm(omega, um.unsqueeze(-1)).squeeze(-1))
        outer_ru = torch.bmm(rm.unsqueeze(-1), um.unsqueeze(1)) / scale_d
        omega = decay * omega + 0.5 * lr * (outer_ru - outer_ru.transpose(1, 2))
        strain = 0.5 * (outer_ru + outer_ru.transpose(1, 2))
        strains.append((scale_d ** 2) * strain.detach().pow(2).sum(dim=-1))
    u_last = z_sub_ctx[:, -1, :] / (z_sub_ctx[:, -1, :].norm(dim=-1, keepdim=True) + 1e-6)
    vort_final = scale_d * torch.bmm(omega, u_last.unsqueeze(-1)).squeeze(-1)
    s_var = torch.stack(strains, dim=0).mean(dim=0)
    nu_eddy = nu_0 + (cs ** 2) * torch.sqrt(s_var + 1e-8)
    nu_mean = nu_eddy.mean(dim=-1, keepdim=True) + 1e-6
    eddy_unit = torch.sqrt((nu_eddy + 0.8 * nu_mean) / (1.8 * nu_mean))
    return vort_final, eddy_unit / (eddy_unit.mean(dim=-1, keepdim=True) + 1e-6), omega

# --- 1. Delta-JEPA: Fast-Weight Causal Delta-Rule + Fisher Preconditioning ---

class DeltaJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, predictor_layers: int = 2, ema_decay: float = 0.995, dropout: float = 0.10, num_subblocks: int = 4, delta_lr: float = 0.25, delta_decay: float = 0.85):
        super().__init__()
        _init_jepa_encoders(self, context_encoder, latent_dim, ema_decay)
        self.num_subblocks, self.delta_lr, self.delta_decay = num_subblocks, delta_lr, delta_decay
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.fast_gate = nn.Parameter(torch.tensor(0.25))
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)

    def _rollout_delta(self, z_ctx_full, z_sub_ctx, z_sub_tgt):
        B, M, D = z_sub_ctx.shape
        W = torch.zeros(B, D, D, device=z_sub_ctx.device, dtype=z_sub_ctx.dtype)
        sq_inns = []
        for m in range(M - 1):
            zm, zn_tgt = z_sub_ctx[:, m, :], z_sub_tgt[:, m + 1, :]
            um = zm / (zm.norm(dim=-1, keepdim=True) + 1e-6)
            rm = zn_tgt - (F.layer_norm(self.predictor(zm), (D,)) + torch.bmm(W, um.unsqueeze(-1)).squeeze(-1))
            sq_inns.append(rm.detach().pow(2))
            W = self.delta_decay * W + self.delta_lr * torch.bmm(rm.unsqueeze(-1), um.unsqueeze(1))
        u_last = z_sub_ctx[:, -1, :] / (z_sub_ctx[:, -1, :].norm(dim=-1, keepdim=True) + 1e-6)
        z_target_pred = F.layer_norm(self.predictor(z_ctx_full) + torch.tanh(self.fast_gate) * torch.bmm(W, u_last.unsqueeze(-1)).squeeze(-1), (D,))
        var_ctx = torch.stack(sq_inns, dim=0).mean(dim=0)
        mean_var = var_ctx.mean(dim=-1, keepdim=True) + 1e-4
        rel_std = torch.sqrt((var_ctx + 0.5 * mean_var) / (1.5 * mean_var))
        return z_target_pred, rel_std / (rel_std.mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx = self.context_encoder(ctx_win)
        z_sub_ctx, z_sub_tgt = _encode_causal_subblocks(ctx_win, self.num_subblocks, self.latent_dim, self.context_encoder, self.target_encoder)
        z_pred, fisher_unit = self._rollout_delta(z_ctx, z_sub_ctx, z_sub_tgt)
        z_tgt = None
        if tgt_win is not None:
            self.target_encoder.eval()
            with torch.no_grad():
                z_tgt = self.target_encoder(tgt_win)
        return z_ctx, z_tgt, z_pred, fisher_unit

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, z_tgt, z_pred, _ = self.forward(ctx, tgt)
        loss = jepa_vicreg_loss(z_pred, z_tgt, z_ctx, sim_weight=1.0, var_weight=1.0, cov_weight=0.5, gamma=0.75, use_log_det=False)
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        _, z_tgt, z_pred, fisher_unit = self.forward(ctx, tgt)
        return self.saliency_gate((z_tgt - z_pred) / fisher_unit)

# --- 2. Speculative-Verification JEPA: Multi-Horizon Draft & Parallel Verification ---

class SpeculativeJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, ema_decay: float = 0.995, dropout: float = 0.10, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay = latent_dim, ema_decay
        self.num_horizons = 3
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        _init_tap_heads(self, context_encoder, latent_dim)
        self.base_predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=2, dropout=0.0)
        self.draft_step = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=2, dropout=0.0)
        self.horizon_embeddings = nn.Parameter(torch.randn(self.num_horizons, latent_dim) * 0.02)
        _init_multihorizon_alien_gates_and_buffers(self, latent_dim, self.num_horizons, coreset_size, top_k_prototypes, null_dim)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        super().update_target_encoder(momentum)
        _update_tap_ema(self, momentum)

    def _draft_horizons(self, z_ctx: torch.Tensor) -> torch.Tensor:
        d = (self.latent_dim,)
        z_full = F.layer_norm(self.base_predictor(z_ctx + self.horizon_embeddings[2].unsqueeze(0)), d)
        z_h0 = F.layer_norm(self.draft_step(z_ctx + self.horizon_embeddings[0].unsqueeze(0)), d)
        z_h1 = F.layer_norm(self.draft_step(z_h0 + self.horizon_embeddings[1].unsqueeze(0)), d)
        return torch.stack([z_h0, z_h1, z_full], dim=1)

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx = _encode_with_multidepth_taps(self.context_encoder, ctx_win, self.context_tap_head)
        z_drafts = self._draft_horizons(z_ctx)
        z_targets = _verify_3horizons(self, tgt_win) if tgt_win is not None else None
        return z_ctx, z_targets, z_drafts

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, z_targets, z_drafts = self.forward(ctx, tgt)
        total_loss = _multihorizon_vicreg_loss(z_drafts, z_targets, z_ctx)
        return total_loss, {"loss": float(total_loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        self.eval()
        fit_covariance_batched(self, train_windows[:, :context_len], train_windows[:, context_len:], residual_fn=lambda c, t: (self.forward(c, t)[1] - self.forward(c, t)[2]).reshape(c.shape[0], -1), dim=self.num_horizons * self.latent_dim, batch_size=batch_size, reg=1e-3, precision_buffer=self.precision_matrix, residual_mean_buffer=self.residual_mean, fitted_buffer=self.precision_fitted)
        J, R = _collect_alien_joints_resids(self, train_windows, context_len, batch_size, is_speculative=True)
        _fit_coreset_and_null_projector(J, R, self.coreset_size, self.top_k_prototypes, self.null_dim, self.joint_center, self.null_projector, self.null_ratio_med, self.coreset_bank, self.unavail_med, self.unavail_mad, self.bank_fitted)

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, use_alien_bank: bool = True, **kwargs) -> torch.Tensor:
        self.eval()
        z_ctx, z_targets, z_drafts = self.forward(ctx, tgt)
        B, H, D = z_drafts.shape
        diffs = z_targets - z_drafts
        if bool(self.precision_fitted.item()):
            e_white = whiten_residuals(diffs.reshape(B, H * D) - self.residual_mean, self.precision_matrix)
            s_joint = self.joint_saliency_gate(e_white)
            d_h = self.saliency_gate(e_white.reshape(B * H, D)).reshape(B, H)
        else:
            d_h = self.saliency_gate(diffs.reshape(B * H, D)).reshape(B, H)
            s_joint = d_h.mean(dim=-1)
        s_spec = torch.sqrt(0.5 * (s_joint.pow(2) + d_h.pow(2).amax(dim=-1)))
        if not use_alien_bank:
            return s_spec
        return _apply_bank_gain(self, z_ctx, z_targets[:, 2, :], z_targets[:, 2, :] - z_drafts[:, 2, :], s_spec, 0.15, 0.10)

# --- 3. Alien Dual-Tower JEPA: Dynamical Coherence + DPP Coreset Availability ---

class AlienDualTowerJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, predictor_layers: int = 2, ema_decay: float = 0.995, dropout: float = 0.10, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay = latent_dim, ema_decay
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)
        _register_alien_bank(self, latent_dim, coreset_size, top_k_prototypes, null_dim)

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx = self.context_encoder(ctx_win)
        z_pred = F.layer_norm(self.predictor(z_ctx), (self.latent_dim,))
        z_tgt = None
        if tgt_win is not None:
            self.target_encoder.eval()
            with torch.no_grad():
                z_tgt = self.target_encoder(tgt_win)
        return z_ctx, z_tgt, z_pred

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, z_tgt, z_pred = self.forward(ctx, tgt)
        loss = jepa_vicreg_loss(z_pred, z_tgt, z_ctx, sim_weight=1.0, var_weight=1.0, cov_weight=0.5, gamma=0.75, use_log_det=False)
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        _fit_alien_bank_default(self, train_windows, context_len, batch_size)

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        z_ctx, z_tgt, z_pred = self.forward(ctx, tgt)
        diff = z_tgt - z_pred
        return _apply_bank_gain(self, z_ctx, z_tgt, diff, self.saliency_gate(diff), 0.20, 0.15)

# --- 4. Eddy-Delta JEPA: Solenoidal Vorticity & Smagorinsky Fast Weights on S^{D-1} ---

class EddyDeltaJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, predictor_layers: int = 2, ema_decay: float = 0.995, dropout: float = 0.10, num_subblocks: int = 4, vorticity_lr: float = 0.25, vorticity_decay: float = 0.85, smagorinsky_cs: float = 0.16, nu_0: float = 1e-3, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay, self.num_subblocks = latent_dim, ema_decay, num_subblocks
        self.num_horizons = 3
        self.vorticity_lr, self.vorticity_decay, self.smagorinsky_cs, self.nu_0 = vorticity_lr, vorticity_decay, smagorinsky_cs, nu_0
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        _init_tap_heads(self, context_encoder, latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.draft_step = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.horizon_embeddings = nn.Parameter(0.02 * torch.randn(self.num_horizons, latent_dim))
        self.vorticity_gate = nn.Parameter(torch.tensor(0.35))
        _init_multihorizon_alien_gates_and_buffers(self, latent_dim, self.num_horizons, coreset_size, top_k_prototypes, null_dim)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        super().update_target_encoder(momentum)
        _update_tap_ema(self, momentum)

    @staticmethod
    def _pure_solenoidal_vorticity(z_sub: torch.Tensor, decay: float, lr: float, cs: float, nu_0: float):
        B, M, D = z_sub.shape
        scale_d = math.sqrt(float(D))
        omega = torch.zeros(B, D, D, device=z_sub.device, dtype=z_sub.dtype)
        strains = []
        for m in range(M - 1):
            zm, zn = z_sub[:, m, :], z_sub[:, m + 1, :].detach()
            um = zm / (zm.norm(dim=-1, keepdim=True) + 1e-6)
            rm = zn - (zm + scale_d * torch.bmm(omega, um.unsqueeze(-1)).squeeze(-1))
            outer = torch.bmm(rm.unsqueeze(-1), um.unsqueeze(1)) / scale_d
            omega = decay * omega + 0.5 * lr * (outer - outer.transpose(1, 2))
            strain = 0.5 * (outer + outer.transpose(1, 2))
            strains.append((scale_d ** 2) * strain.detach().pow(2).sum(dim=-1))
        u_last = z_sub[:, -1, :] / (z_sub[:, -1, :].norm(dim=-1, keepdim=True) + 1e-6)
        vort_final = scale_d * torch.bmm(omega, u_last.unsqueeze(-1)).squeeze(-1)
        s_var = torch.stack(strains, dim=0).mean(dim=0)
        nu_eddy = nu_0 + (cs ** 2) * torch.sqrt(s_var + 1e-8)
        nu_mean = nu_eddy.mean(dim=-1, keepdim=True) + 1e-6
        eddy_unit = torch.sqrt((nu_eddy + 0.8 * nu_mean) / (1.8 * nu_mean))
        return vort_final, eddy_unit / (eddy_unit.mean(dim=-1, keepdim=True) + 1e-6), omega

    def _forward_multihorizon(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        d = (self.latent_dim,)
        z_ctx = _encode_with_multidepth_taps(self.context_encoder, ctx_win, self.context_tap_head)
        z_sub_ctx, _ = _encode_causal_subblocks(ctx_win, self.num_subblocks, self.latent_dim, self.context_encoder, self.target_encoder)
        vort_final, eddy_unit, omega = self._pure_solenoidal_vorticity(z_sub_ctx, self.vorticity_decay, self.vorticity_lr, self.smagorinsky_cs, self.nu_0)
        v_g = 0.20 * torch.tanh(self.vorticity_gate)
        z_h0 = F.layer_norm(self.draft_step(z_ctx + self.horizon_embeddings[0].unsqueeze(0)) + 0.5 * v_g * vort_final, d)
        z_h1 = F.layer_norm(self.draft_step(z_h0 + self.horizon_embeddings[1].unsqueeze(0)) + 0.75 * v_g * vort_final, d)
        z_h2 = F.layer_norm(self.predictor(z_ctx + self.horizon_embeddings[2].unsqueeze(0)) + v_g * vort_final, d)
        z_drafts = torch.stack([z_h0, z_h1, z_h2], dim=1)
        z_targets = _verify_3horizons(self, tgt_win) if tgt_win is not None else None
        return z_ctx, z_targets, z_drafts, eddy_unit, omega

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx, z_targets, z_drafts, eddy_unit, omega = self._forward_multihorizon(ctx_win, tgt_win)
        z_tgt = z_targets[:, 2, :] if z_targets is not None else None
        return z_ctx, z_tgt, z_drafts[:, 2, :], eddy_unit, omega

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, z_targets, z_drafts = self._forward_multihorizon(ctx, tgt)[:3]
        loss = _multihorizon_vicreg_loss(z_drafts, z_targets, z_ctx)
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        _fit_multihorizon_alien_pipeline(self, train_windows, context_len, batch_size, residual_fn=lambda c, t: (self._forward_multihorizon(c, t)[1] - self._forward_multihorizon(c, t)[2]).reshape(c.shape[0], -1))

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, use_alien_bank: bool = True, **kwargs) -> torch.Tensor:
        self.eval()
        z_ctx, z_targets, z_drafts, eddy_unit, _ = self._forward_multihorizon(ctx, tgt)
        B, H, D = z_drafts.shape
        diffs = z_targets - z_drafts
        shear_gain = torch.sqrt(eddy_unit).unsqueeze(1)
        if bool(self.precision_fitted.item()):
            e_white = whiten_residuals(diffs.reshape(B, H * D) - self.residual_mean, self.precision_matrix)
            e_sc = (e_white.reshape(B, H, D) * shear_gain).reshape(B, H * D)
            s_joint = self.joint_saliency_gate(e_sc)
            d_h = self.saliency_gate(e_sc.reshape(B * H, D)).reshape(B, H)
        else:
            d_h = self.saliency_gate((diffs * shear_gain).reshape(B * H, D)).reshape(B, H)
            s_joint = d_h.mean(dim=-1)
        s_eddy = torch.sqrt(0.5 * (s_joint.pow(2) + d_h.pow(2).amax(dim=-1)))
        if not use_alien_bank:
            return s_eddy
        return _apply_bank_gain(self, z_ctx, z_targets[:, 2, :], diffs[:, 2, :], s_eddy, 0.15, 0.10)

# --- 5. Anchored-Stokes JEPA: Halpern Monotone Advection-Diffusion on S^{D-1} ---

class MonotoneStokesOperator(nn.Module):
    """Coercive advection-diffusion operator M_NS = 0.5(A - A^T) + B B^T + nu_0 I."""

    def __init__(self, latent_dim: int, nu_0: float = 1e-2):
        super().__init__()
        self.latent_dim, self.nu_0 = latent_dim, nu_0
        scale = 0.5 / math.sqrt(float(latent_dim))
        self.skew_gen = nn.Parameter(scale * torch.randn(latent_dim, latent_dim))
        self.diss_gen = nn.Parameter(scale * torch.randn(latent_dim, latent_dim))

    def operator_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.latent_dim, device=self.skew_gen.device, dtype=self.skew_gen.dtype)
        m_op = 0.5 * (self.skew_gen - self.skew_gen.T) + (self.diss_gen @ self.diss_gen.T) + self.nu_0 * eye
        k_lip = torch.sqrt(m_op.abs().sum(dim=0).max() * m_op.abs().sum(dim=1).max()) + 1e-4
        return m_op, k_lip

    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        m_op, _ = self.operator_matrix()
        return F.linear(delta_z, m_op)

    def rollout_halpern(self, z_ctx: torch.Tensor, z_eq: torch.Tensor, steps: int, gamma: float) -> torch.Tensor:
        _, k_lip = self.operator_matrix()
        z_0, z_k = z_ctx, z_ctx
        for k in range(1, steps + 1):
            t_val = float(k) + gamma
            alpha_k = 1.0 / (k_lip * math.sqrt(t_val))
            beta_k = gamma / t_val
            z_k = z_k - alpha_k * self(z_k - z_eq) + beta_k * (z_0 - z_k)
        return z_k - z_0

class AnchoredStokesJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, ema_decay: float = 0.995, dropout: float = 0.10, rollout_steps: int = 4, anchor_gamma: float = 2.0, nu_0: float = 1e-2, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay, self.rollout_steps, self.anchor_gamma, self.nu_0 = latent_dim, ema_decay, rollout_steps, anchor_gamma, nu_0
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        self.equilibrium_head = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=2, dropout=0.0)
        self.stokes_op = MonotoneStokesOperator(latent_dim=latent_dim, nu_0=nu_0)
        self.rollout_gate = nn.Parameter(torch.tensor(0.65))
        self.saliency_gate = CoordinateSaliencyGate(dim=latent_dim, tau=0.5, alpha=0.5)
        _register_alien_bank(self, latent_dim, coreset_size, top_k_prototypes, null_dim)

    def apply_monotone_operator(self, z: torch.Tensor, z_eq: torch.Tensor) -> torch.Tensor:
        return self.stokes_op(z - z_eq)

    def rollout_anchored(self, z_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = (self.latent_dim,)
        z_eq = F.layer_norm(self.equilibrium_head(z_ctx), d)
        disp = self.stokes_op.rollout_halpern(z_ctx, z_eq, self.rollout_steps, self.anchor_gamma)
        z_pred = F.layer_norm(z_eq + 0.15 * torch.tanh(self.rollout_gate) * disp, d)
        return z_pred, z_eq

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx = self.context_encoder(ctx_win)
        z_pred, z_eq = self.rollout_anchored(z_ctx)
        z_tgt = None
        if tgt_win is not None:
            self.target_encoder.eval()
            with torch.no_grad():
                z_tgt = self.target_encoder(tgt_win)
        return z_ctx, z_tgt, z_pred, z_eq

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, z_tgt, z_pred, _ = self.forward(ctx, tgt)
        loss = jepa_vicreg_loss(z_pred, z_tgt, z_ctx, sim_weight=1.0, var_weight=1.0, cov_weight=0.5, gamma=0.75, use_log_det=False)
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        _fit_alien_bank_default(self, train_windows, context_len, batch_size)

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        z_ctx, z_tgt, z_pred, _ = self.forward(ctx, tgt)
        diff = z_tgt - z_pred
        _, k_lip = self.stokes_op.operator_matrix()
        s_stokes = 0.90 * self.saliency_gate(diff) + 0.10 * self.saliency_gate(self.stokes_op(diff) / k_lip)
        return _apply_bank_gain(self, z_ctx, z_tgt, diff, s_stokes, 0.22, 0.16)

# --- 6. Soliton-Reynolds JEPA: Lieb-Thirring Soliton Concentration on S^{D-1} ---

class SolitonReynoldsJEPAModel(JEPABase):
    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, predictor_layers: int = 2, ema_decay: float = 0.995, dropout: float = 0.10, lt_gamma: float = 1.0, num_subblocks: int = 4, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay, self.lt_gamma, self.num_subblocks = latent_dim, ema_decay, lt_gamma, num_subblocks
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        _init_tap_heads(self, context_encoder, latent_dim)
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.sub_predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.subblock_embeddings = nn.Parameter(0.02 * torch.randn(num_subblocks, latent_dim))
        _init_multihorizon_alien_gates_and_buffers(self, latent_dim, num_subblocks, coreset_size, top_k_prototypes, null_dim)
        self.register_buffer("r_lt_med", torch.tensor(1.0 / max(num_subblocks - 1, 1)))
        self.register_buffer("r_lt_mad", torch.tensor(0.05))

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        super().update_target_encoder(momentum)
        _update_tap_ema(self, momentum)

    @staticmethod
    def lieb_thirring_soliton_ratio(sub_residuals: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        gram = torch.bmm(sub_residuals, sub_residuals.transpose(1, 2))
        evals = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(1, 2))).clamp(min=0.0)
        moments = evals.pow(gamma + 0.5)
        return moments[:, -1] / (moments.sum(dim=-1) + 1e-12)

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        M, D = self.num_subblocks, self.latent_dim
        d = (D,)
        z_ctx = _encode_with_multidepth_taps(self.context_encoder, ctx_win, self.context_tap_head)
        horizons = []
        z_prev = z_ctx
        for m in range(M - 1):
            z_prev = F.layer_norm(self.sub_predictor(z_prev + self.subblock_embeddings[m].unsqueeze(0)), d)
            horizons.append(z_prev)
        z_pred = F.layer_norm(self.predictor(z_ctx + self.subblock_embeddings[-1].unsqueeze(0)), d)
        horizons.append(z_pred)
        z_sub_pred = torch.stack(horizons, dim=1)
        z_tgt, z_sub_tgt = None, None
        if tgt_win is not None:
            S_len = tgt_win.shape[1]
            self.target_encoder.eval()
            self.target_tap_head.eval()
            with torch.no_grad():
                t_list = [_encode_with_multidepth_taps(self.target_encoder, tgt_win[:, :(S_len if m == M - 1 else max(((m + 1) * S_len) // M, 2)), :], self.target_tap_head) for m in range(M)]
            z_sub_tgt = torch.stack(t_list, dim=1)
            z_tgt = z_sub_tgt[:, -1, :]
        return z_ctx, z_tgt, z_pred, z_sub_pred, z_sub_tgt

    def _subblock_residuals_flat(self, ctx_win: torch.Tensor, tgt_win: torch.Tensor) -> torch.Tensor:
        _, _, _, zsp, zst = self.forward(ctx_win, tgt_win)
        return (zst - zsp).reshape(ctx_win.shape[0], -1)

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, _, _, zsp, zst = self.forward(ctx, tgt)
        M = self.num_subblocks
        w = [0.50 / max(M - 1, 1)] * (M - 1) + [0.50]
        loss = sum(wh * jepa_vicreg_loss(zsp[:, h, :], zst[:, h, :], z_ctx if h == M - 1 else None, use_log_det=True) for h, wh in enumerate(w))
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        self.eval()
        M, D = self.num_subblocks, self.latent_dim
        ctx_all, tgt_all = train_windows[:, :context_len], train_windows[:, context_len:]
        fit_covariance_batched(
            self, ctx_all, tgt_all, residual_fn=self._subblock_residuals_flat,
            dim=M * D, batch_size=batch_size, reg=1e-3,
            precision_buffer=self.precision_matrix, residual_mean_buffer=self.residual_mean, fitted_buffer=self.precision_fitted,
        )
        device = next(self.parameters()).device
        r_lts = []
        for i in range(0, len(train_windows), batch_size):
            b = torch.from_numpy(train_windows[i : i + batch_size]).float().to(device)
            res_flat = self._subblock_residuals_flat(b[:, :context_len], b[:, context_len:])
            ew = whiten_residuals(res_flat - self.residual_mean, self.precision_matrix).reshape(len(b), M, D)
            r_lts.append(self.lieb_thirring_soliton_ratio(ew - ew.mean(dim=1, keepdim=True), gamma=self.lt_gamma))
        all_r = torch.cat(r_lts, dim=0)
        med = torch.median(all_r)
        self.r_lt_med.copy_(med)
        self.r_lt_mad.copy_(torch.median(torch.abs(all_r - med)).clamp(min=1e-3))

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        _, _, _, z_sub_pred, z_sub_tgt = self.forward(ctx, tgt)
        B, M, D = z_sub_pred.shape
        diffs = z_sub_tgt - z_sub_pred
        if bool(self.precision_fitted.item()):
            e_white = whiten_residuals(diffs.reshape(B, M * D) - self.residual_mean, self.precision_matrix)
            s_joint = self.joint_saliency_gate(e_white)
            d_h = self.saliency_gate(e_white.reshape(B * M, D)).reshape(B, M)
            fluct = e_white.reshape(B, M, D) - e_white.reshape(B, M, D).mean(dim=1, keepdim=True)
        else:
            d_h = self.saliency_gate(diffs.reshape(B * M, D)).reshape(B, M)
            s_joint = d_h.mean(dim=-1)
            fluct = diffs - diffs.mean(dim=1, keepdim=True)
        s_env = torch.sqrt(0.5 * (s_joint.pow(2) + d_h.pow(2).amax(dim=-1)))
        if not kwargs.get("use_soliton", True):
            return s_env
        alpha_sol = float(kwargs.get("alpha_sol", 0.12))
        r_lt = self.lieb_thirring_soliton_ratio(fluct, gamma=self.lt_gamma)
        z_lt = (r_lt - self.r_lt_med) / (1.4826 * self.r_lt_mad)
        return s_env * (1.0 + alpha_sol * F.relu(torch.tanh(0.5 * z_lt)))

# --- 7. Unified Helmholtz-Soliton JEPA: Solenoidal-Stokes + Soliton Wavepacket ---

class HelmholtzSolitonJEPAModel(JEPABase):
    """Unified Multi-Horizon Helmholtz-Hodge Solenoidal-Stokes & Lieb-Thirring Soliton Fluid JEPA on S^{D-1}."""

    def __init__(self, context_encoder: nn.Module, latent_dim: int = 32, predictor_hidden_dim: int = 64, predictor_layers: int = 2, ema_decay: float = 0.995, dropout: float = 0.10, num_subblocks: int = 4, rollout_steps: int = 4, anchor_gamma: float = 2.0, vorticity_lr: float = 0.25, vorticity_decay: float = 0.85, smagorinsky_cs: float = 0.16, nu_0: float = 1e-2, lt_gamma: float = 1.0, coreset_size: int = 64, top_k_prototypes: int = 5, null_dim: int = 8):
        super().__init__()
        _disable_dropout(context_encoder)
        self.context_encoder = context_encoder
        self.latent_dim, self.ema_decay, self.num_subblocks = latent_dim, ema_decay, num_subblocks
        self.num_horizons = 3
        self.rollout_steps, self.anchor_gamma = rollout_steps, anchor_gamma
        self.vorticity_lr, self.vorticity_decay, self.smagorinsky_cs, self.nu_0, self.lt_gamma = vorticity_lr, vorticity_decay, smagorinsky_cs, nu_0, lt_gamma
        self.target_encoder = self.init_target_encoder(context_encoder)
        _disable_dropout(self.target_encoder)
        _init_tap_heads(self, context_encoder, latent_dim)
        self.equilibrium_head = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.sub_predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim, num_layers=predictor_layers, dropout=0.0)
        self.subblock_embeddings = nn.Parameter(0.02 * torch.randn(self.num_horizons, latent_dim))
        self.stokes_op = MonotoneStokesOperator(latent_dim=latent_dim, nu_0=nu_0)
        self.vorticity_gate = nn.Parameter(torch.tensor(0.25))
        self.stokes_gate = nn.Parameter(torch.tensor(0.35))
        _init_multihorizon_alien_gates_and_buffers(self, latent_dim, self.num_horizons, coreset_size, top_k_prototypes, null_dim)

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        super().update_target_encoder(momentum)
        _update_tap_ema(self, momentum)

    def _draft_fluid_horizons(self, z_ctx: torch.Tensor, vort_final: torch.Tensor) -> torch.Tensor:
        d = (self.latent_dim,)
        s_g = 0.15 * torch.tanh(self.stokes_gate)
        v_g = 0.15 * torch.tanh(self.vorticity_gate)
        z_eq0 = F.layer_norm(self.sub_predictor(z_ctx + self.subblock_embeddings[0].unsqueeze(0)), d)
        z_h0 = F.layer_norm(z_eq0 + s_g * self.stokes_op.rollout_halpern(z_ctx, z_eq0, 2, self.anchor_gamma) + 0.5 * v_g * vort_final, d)
        z_eq1 = F.layer_norm(self.sub_predictor(z_h0 + self.subblock_embeddings[1].unsqueeze(0)), d)
        z_h1 = F.layer_norm(z_eq1 + s_g * self.stokes_op.rollout_halpern(z_h0, z_eq1, 3, self.anchor_gamma) + 0.75 * v_g * vort_final, d)
        z_eq2 = F.layer_norm(self.equilibrium_head(z_ctx + self.subblock_embeddings[2].unsqueeze(0)), d)
        z_h2 = F.layer_norm(z_eq2 + s_g * self.stokes_op.rollout_halpern(z_ctx, z_eq2, self.rollout_steps, self.anchor_gamma) + v_g * vort_final, d)
        return torch.stack([z_h0, z_h1, z_h2], dim=1)

    def forward(self, ctx_win: torch.Tensor, tgt_win: Optional[torch.Tensor] = None):
        z_ctx = _encode_with_multidepth_taps(self.context_encoder, ctx_win, self.context_tap_head)
        z_sub_ctx, z_sub_tgt_ctx = _encode_causal_subblocks(ctx_win, self.num_subblocks, self.latent_dim, self.context_encoder, self.target_encoder)
        vort_final, eddy_unit, _ = _rollout_solenoidal_vorticity(self.equilibrium_head, z_sub_ctx, z_sub_tgt_ctx, self.vorticity_decay, self.vorticity_lr, self.smagorinsky_cs, self.nu_0)
        z_sub_pred = self._draft_fluid_horizons(z_ctx, vort_final)
        z_pred = z_sub_pred[:, 2, :]
        z_tgt, z_sub_tgt = None, None
        if tgt_win is not None:
            z_sub_tgt = _verify_3horizons(self, tgt_win)
            z_tgt = z_sub_tgt[:, 2, :]
        return z_ctx, z_tgt, z_pred, eddy_unit, z_sub_pred, z_sub_tgt

    def compute_objective(self, ctx: torch.Tensor, tgt: torch.Tensor, config=None, **kwargs):
        z_ctx, _, _, _, z_sub_pred, z_sub_tgt = self.forward(ctx, tgt)
        total_loss = _multihorizon_vicreg_loss(z_sub_pred, z_sub_tgt, z_ctx)
        return total_loss, {"loss": float(total_loss.item())}

    @torch.no_grad()
    def fit_alien_bank(self, train_windows, context_len: int = 256, batch_size: int = 256) -> None:
        self.eval()
        fit_covariance_batched(self, train_windows[:, :context_len], train_windows[:, context_len:], residual_fn=lambda c, t: (self.forward(c, t)[5] - self.forward(c, t)[4]).reshape(c.shape[0], -1), dim=self.num_horizons * self.latent_dim, batch_size=batch_size, reg=1e-3, precision_buffer=self.precision_matrix, residual_mean_buffer=self.residual_mean, fitted_buffer=self.precision_fitted)
        J, R = _collect_alien_joints_resids(self, train_windows, context_len, batch_size, is_speculative=False)
        _fit_coreset_and_null_projector(J, R, self.coreset_size, self.top_k_prototypes, self.null_dim, self.joint_center, self.null_projector, self.null_ratio_med, self.coreset_bank, self.unavail_med, self.unavail_mad, self.bank_fitted)

    @torch.no_grad()
    def compute_predictive_discrepancy(self, ctx: torch.Tensor, tgt: torch.Tensor, use_alien_bank: bool = True, **kwargs) -> torch.Tensor:
        self.eval()
        z_ctx, z_tgt, z_pred, _, z_sub_pred, z_sub_tgt = self.forward(ctx, tgt)
        B, H, D = z_sub_pred.shape
        diffs = z_sub_tgt - z_sub_pred
        if bool(self.precision_fitted.item()):
            e_white = whiten_residuals(diffs.reshape(B, H * D) - self.residual_mean, self.precision_matrix)
            s_joint = self.joint_saliency_gate(e_white)
            d_h = self.saliency_gate(e_white.reshape(B * H, D)).reshape(B, H)
            fluct = e_white.reshape(B, H, D) - e_white.reshape(B, H, D).mean(dim=1, keepdim=True)
        else:
            d_h = self.saliency_gate(diffs.reshape(B * H, D)).reshape(B, H)
            s_joint = d_h.mean(dim=-1)
            fluct = diffs - diffs.mean(dim=1, keepdim=True)
        s_env = torch.sqrt(0.5 * (s_joint.pow(2) + d_h.pow(2).amax(dim=-1)))
        r_lt = SolitonReynoldsJEPAModel.lieb_thirring_soliton_ratio(fluct, gamma=self.lt_gamma)
        soliton_gain = 1.0 + 0.08 * F.relu(r_lt - (1.0 / float(H)))
        s_fluid = s_env * soliton_gain
        if not use_alien_bank:
            return s_fluid
        return _apply_bank_gain(self, z_ctx, z_tgt, z_tgt - z_pred, s_fluid, 0.15, 0.10)

# Exported Aliases
DeltaJEPA = DeltaJEPAModel
SpeculativeJEPA = SpeculativeJEPAModel
AlienDualTowerJEPA = AlienDualTowerJEPAModel
EddyDeltaJEPA = EddyDeltaJEPAModel
AnchoredStokesJEPA = AnchoredStokesJEPAModel
SolitonReynoldsJEPA = SolitonReynoldsJEPAModel
HelmholtzSolitonJEPA = HelmholtzSolitonJEPAModel
