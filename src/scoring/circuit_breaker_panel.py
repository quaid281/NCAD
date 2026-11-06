"""224-Bit Multi-Perspective Circuit-Breaker Panel (Zero 1D Score Collapse, Zero SPOT).

Eliminates premature 1D scalar score collapse and Extreme Value Theory (SPOT) scalar
thresholding by monitoring 7 physical-perspective binary circuit breakers per latent
coordinate (7 x 32 = 224 switches) plus 32 fast right-edge onset relays:
  1. 128 Sub-block position switches |e_{m,c}| across 4 prefix horizons (16, 32, 48, 64 steps)
  2.  32 Slow-drift switches |mean_m e_{m,c}|
  3.  32 Turbulent-fluctuation switches std_m(e_{m,c})
  4.  32 Right-edge acceleration switches |e_{3,c} - e_{0,c}|
  5.  32 Fast right-edge onset relays (m=3 AND (right-edge acceleration OR fluctuation OR m=2))
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from src.models._jepa_utils import whiten_residuals


def build_224_perspective_dials(ew_signed: np.ndarray) -> np.ndarray:
    """Expand (N_win, 4, D) signed multi-horizon residuals into (N_win, 7*D) perspective dials."""
    n_win, n_sub, dim = ew_signed.shape
    if n_sub != 4:
        raise ValueError(f"Expected 4 sub-blocks along axis=1, got shape {ew_signed.shape}.")
    d_sub = np.abs(ew_signed).reshape(n_win, 4 * dim)
    e_bar = ew_signed.mean(axis=1)
    d_drift = np.abs(e_bar)
    d_fluct = np.sqrt(np.mean((ew_signed - e_bar[:, None, :]) ** 2, axis=1))
    d_accel = np.abs(ew_signed[:, 3, :] - ew_signed[:, 0, :])
    return np.concatenate([d_sub, d_drift, d_fluct, d_accel], axis=1)


def stabilize_whitened_subblocks(
    va_white4: np.ndarray,
    te_white4: np.ndarray,
    va_raw4: np.ndarray,
    te_raw4: np.ndarray,
    n_cal_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend whitened and raw sub-block residuals per coordinate using nominal calibration stability.

    On training data, Mahalanobis-whitened coordinates have unit variance. If a whitened
    coordinate exhibits RMS > 1.10 on held-out nominal calibration windows `[:n_cal_w]`,
    the precision matrix amplified out-of-sample drift along that direction, so the panel
    smoothly falls back toward the calibration-normalized unwhitened residual.
    """
    va_w = np.clip(np.nan_to_num(va_white4.astype(np.float64), nan=0.0), -100.0, 100.0)
    te_w = np.clip(np.nan_to_num(te_white4.astype(np.float64), nan=0.0), -100.0, 100.0)
    va_r = np.clip(np.nan_to_num(va_raw4.astype(np.float64), nan=0.0), -100.0, 100.0)
    te_r = np.clip(np.nan_to_num(te_raw4.astype(np.float64), nan=0.0), -100.0, 100.0)

    rms_white_cal = np.sqrt(np.mean(va_w[:n_cal_w] ** 2, axis=0, keepdims=True)) + 1e-4
    rms_raw_cal = np.sqrt(np.mean(va_r[:n_cal_w] ** 2, axis=0, keepdims=True)) + 1e-4
    raw_norm_va = va_r / rms_raw_cal
    raw_norm_te = te_r / rms_raw_cal

    overfit_tau = np.maximum(0.0, rms_white_cal - 1.10) ** 2
    w_white = 1.0 / (1.0 + 2.0 * overfit_tau)
    return (
        (w_white * va_w + (1.0 - w_white) * raw_norm_va).astype(np.float64),
        (w_white * te_w + (1.0 - w_white) * raw_norm_te).astype(np.float64),
    )


def aggregate_vector_stream(
    vecs: np.ndarray,
    n_points: int,
    context_size: int,
    suspect_size: int,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window-level switch vectors onto the timeline via trailing or uniform smear."""
    n_win, dim = vecs.shape
    out = np.zeros((n_points, dim), dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.float64)

    if mapping == "trailing":
        idx = np.arange(n_win) + context_size + suspect_size - 1
        valid = idx < n_points
        out[idx[valid]] = vecs[valid]
        counts[idx[valid]] = 1.0
    elif mapping == "smear":
        diff_arr = np.zeros((n_points + 1, dim), dtype=np.float64)
        diff_cnt = np.zeros(n_points + 1, dtype=np.float64)
        starts = np.arange(n_win) + context_size
        ends = np.minimum(starts + suspect_size, n_points)
        valid = (starts < n_points) & (ends > starts)
        np.add.at(diff_arr, starts[valid], vecs[valid])
        np.add.at(diff_arr, ends[valid], -vecs[valid])
        np.add.at(diff_cnt, starts[valid], 1.0)
        np.add.at(diff_cnt, ends[valid], -1.0)
        out = np.cumsum(diff_arr[:-1], axis=0)
        counts = np.cumsum(diff_cnt[:-1])
        mask = counts > 0
        out[mask] /= counts[mask, None]
    else:
        raise ValueError(f"Unsupported mapping '{mapping}'. Expected 'trailing' or 'smear'.")

    return out.astype(np.float32), counts > 0


def aggregate_prefix_224_stream(
    vecs224: np.ndarray,
    n_points: int,
    context_size: int,
    suspect_size: int,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each of the 7 perspective groups onto the exact time steps it physically observes."""
    if mapping == "trailing":
        return aggregate_vector_stream(vecs224, n_points, context_size, suspect_size, "trailing")

    n_win, total_dim = vecs224.shape
    dim = total_dim // 7
    diff_arr = np.zeros((n_points + 1, total_dim), dtype=np.float64)
    diff_cnt = np.zeros((n_points + 1, total_dim), dtype=np.float64)
    starts = np.arange(n_win) + context_size

    q1 = max(1, suspect_size // 4)
    q2 = max(2, suspect_size // 2)
    q3 = max(3, (3 * suspect_size) // 4)
    spans = [
        (0 * dim, 1 * dim, 0, q1),
        (1 * dim, 2 * dim, 0, q2),
        (2 * dim, 3 * dim, 0, q3),
        (3 * dim, 4 * dim, 0, suspect_size),
        (4 * dim, 5 * dim, 0, suspect_size),
        (5 * dim, 6 * dim, 0, suspect_size),
        (6 * dim, 7 * dim, q3, suspect_size),
    ]
    for c0, c1, t0, t1 in spans:
        s_idx = starts + t0
        e_idx = np.minimum(starts + t1, n_points)
        valid = (s_idx < n_points) & (e_idx > s_idx)
        np.add.at(diff_arr[:, c0:c1], s_idx[valid], vecs224[valid, c0:c1])
        np.add.at(diff_arr[:, c0:c1], e_idx[valid], -vecs224[valid, c0:c1])
        np.add.at(diff_cnt[:, c0:c1], s_idx[valid], 1.0)
        np.add.at(diff_cnt[:, c0:c1], e_idx[valid], -1.0)

    out = np.cumsum(diff_arr[:-1], axis=0)
    cnt = np.cumsum(diff_cnt[:-1], axis=0)
    mask = cnt > 0
    out[mask] /= cnt[mask]
    valid_t = cnt[:, 3 * dim] > 0
    return out.astype(np.float32), valid_t


@dataclass
class CircuitBreakerStreamDecision:
    """Outputs of the 224-Bit Circuit-Breaker Panel on a stream timeline."""

    valid_mask: np.ndarray
    cal_switch_counts: np.ndarray
    hold_switch_counts: np.ndarray
    test_switch_counts: np.ndarray
    hold_pred_strict: np.ndarray
    test_pred_strict: np.ndarray
    hold_pred_balanced: np.ndarray
    test_pred_balanced: np.ndarray
    k_gate_strict: float
    k_gate_balanced: float


class MultiPerspectiveCircuitBreakerPanel:
    """Zero-1D-Score, Zero-SPOT 224-Bit Multi-Perspective Circuit-Breaker Panel."""

    def __init__(self, context_size: int = 256, suspect_size: int = 64):
        self.context_size = context_size
        self.suspect_size = suspect_size

    def calibrate_and_trip_windows(
        self, va_ew: np.ndarray, te_ew: np.ndarray, n_cal_w: int
    ) -> dict[str, np.ndarray]:
        """Calibrate per-dial circuit breakers on `va_ew[:n_cal_w]` and return binary window states."""
        n_win_va, _, dim = va_ew.shape
        n_win_te = len(te_ew)
        va_128 = np.abs(va_ew).reshape(n_win_va, 4 * dim)
        te_128 = np.abs(te_ew).reshape(n_win_te, 4 * dim)
        va_224 = build_224_perspective_dials(va_ew)
        te_224 = build_224_perspective_dials(te_ew)

        med_128 = np.median(va_128[:n_cal_w], axis=0, keepdims=True)
        mad_128 = 1.4826 * np.median(np.abs(va_128[:n_cal_w] - med_128), axis=0, keepdims=True) + 1e-4
        z128_va = np.maximum(0.0, (va_128 - med_128) / mad_128)
        z128_te = np.maximum(0.0, (te_128 - med_128) / mad_128)

        med_224 = np.median(va_224[:n_cal_w], axis=0, keepdims=True)
        mad_224 = 1.4826 * np.median(np.abs(va_224[:n_cal_w] - med_224), axis=0, keepdims=True) + 1e-4
        z224_va = np.maximum(0.0, (va_224 - med_224) / mad_224)
        z224_te = np.maximum(0.0, (te_224 - med_224) / mad_224)

        q1 = max(1, n_cal_w // 2)
        drift_128 = np.maximum(
            1.0,
            (np.percentile(z128_va[q1:n_cal_w], 98.0, axis=0, keepdims=True) + 0.5)
            / (np.percentile(z128_va[:q1], 98.0, axis=0, keepdims=True) + 0.5),
        )
        drift_224 = np.maximum(
            1.0,
            (np.percentile(z224_va[q1:n_cal_w], 98.0, axis=0, keepdims=True) + 0.5)
            / (np.percentile(z224_va[:q1], 98.0, axis=0, keepdims=True) + 0.5),
        )

        s_cal_128 = med_128 + mad_128
        tr_shift_128 = np.maximum(1.0, s_cal_128 / (1.5 * np.median(s_cal_128, keepdims=True) + 1e-4))

        s_cal_224 = med_224 + mad_224
        s_ref_224 = (
            np.concatenate(
                [
                    np.repeat(np.median(s_cal_224[:, : 4 * dim], keepdims=True), 4 * dim, axis=1),
                    np.repeat(np.median(s_cal_224[:, 4 * dim : 5 * dim], keepdims=True), dim, axis=1),
                    np.repeat(np.median(s_cal_224[:, 5 * dim : 6 * dim], keepdims=True), dim, axis=1),
                    np.repeat(np.median(s_cal_224[:, 6 * dim : 7 * dim], keepdims=True), dim, axis=1),
                ],
                axis=1,
            )
            + 1e-4
        )
        tr_shift_224 = np.maximum(1.0, s_cal_224 / (1.5 * s_ref_224))

        guard_128 = np.where(drift_128 > 1.35, drift_128, 1.0) * np.where(tr_shift_128 > 1.25, tr_shift_128, 1.0)
        guard_224 = np.where(drift_224 > 1.35, drift_224, 1.0) * np.where(tr_shift_224 > 1.25, tr_shift_224, 1.0)

        tau_128 = (np.percentile(z128_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4) * guard_128
        bits128_va = (z128_va > tau_128).astype(np.float32)
        bits128_te = (z128_te > tau_128).astype(np.float32)

        tau_224 = (np.percentile(z224_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4) * guard_224
        bits224_va = (z224_va > tau_224).astype(np.float32)
        bits224_te = (z224_te > tau_224).astype(np.float32)

        z7_va = z224_va.reshape(-1, 7, dim)
        z7_te = z224_te.reshape(-1, 7, dim)
        g7 = guard_224.reshape(1, 7, dim)
        tau_edge = (np.percentile(z7_va[:n_cal_w], 99.3, axis=0, keepdims=True) + 1e-4) * (g7**1.25)
        b_edge_va = z7_va > tau_edge
        b_edge_te = z7_te > tau_edge
        fast_edge_va = (b_edge_va[:, 3, :] & (b_edge_va[:, 6, :] | b_edge_va[:, 5, :] | b_edge_va[:, 2, :])).astype(
            np.float32
        )
        fast_edge_te = (b_edge_te[:, 3, :] & (b_edge_te[:, 6, :] | b_edge_te[:, 5, :] | b_edge_te[:, 2, :])).astype(
            np.float32
        )

        return {
            "bits128_va": bits128_va,
            "bits128_te": bits128_te,
            "bits224_va": bits224_va,
            "bits224_te": bits224_te,
            "fast_edge_va": fast_edge_va,
            "fast_edge_te": fast_edge_te,
        }

    def evaluate_stream_decisions(
        self,
        panel_bits: dict[str, np.ndarray],
        len_va: int,
        len_test: int,
        mapping: str,
    ) -> CircuitBreakerStreamDecision:
        """Project binary switch states onto the timeline and apply coincidence gates."""
        C, S = self.context_size, self.suspect_size
        vp_pref, vm = aggregate_prefix_224_stream(panel_bits["bits224_va"], len_va, C, S, mapping)
        tp_pref, tm = aggregate_prefix_224_stream(panel_bits["bits224_te"], len_test, C, S, mapping)

        vp_b128, _ = aggregate_vector_stream(panel_bits["bits128_va"], len_va, C, S, mapping)
        tp_b128, _ = aggregate_vector_stream(panel_bits["bits128_te"], len_test, C, S, mapping)

        vp_edge, _ = aggregate_vector_stream(panel_bits["fast_edge_va"], len_va, C, S, mapping)
        tp_edge, _ = aggregate_vector_stream(panel_bits["fast_edge_te"], len_test, C, S, mapping)

        vpref, tpref = vp_pref[vm], tp_pref[tm]
        vb128, tb128 = vp_b128[vm], tp_b128[tm]
        vedge, tedge = vp_edge[vm], tp_edge[tm]
        half = len(vpref) // 2

        tot_va = vpref.sum(axis=1) + vb128.sum(axis=1) + 2.0 * vedge.sum(axis=1)
        tot_te = tpref.sum(axis=1) + tb128.sum(axis=1) + 2.0 * tedge.sum(axis=1)

        k_edge = max(float(np.percentile(vedge[:half].sum(axis=1), 99.90)) * 1.25, 2.5)
        edge_hold = (vedge[half + S :].sum(axis=1) > k_edge).astype(int)
        edge_test = (tedge.sum(axis=1) > k_edge).astype(int)

        k_128_bal = max(float(np.percentile(vb128[:half].sum(axis=1), 99.5)) * 1.12, 3.5)
        k_pref_bal = max(float(np.percentile(vpref[:half].sum(axis=1), 99.5)) * 1.12, 5.0)
        hold_bal = np.maximum(
            np.maximum(
                (vb128[half + S :].sum(axis=1) > k_128_bal).astype(int),
                (vpref[half + S :].sum(axis=1) > k_pref_bal).astype(int),
            ),
            edge_hold,
        )
        test_bal = np.maximum(
            np.maximum(
                (tb128.sum(axis=1) > k_128_bal).astype(int),
                (tpref.sum(axis=1) > k_pref_bal).astype(int),
            ),
            edge_test,
        )

        k_128_strict = max(float(np.percentile(vb128[:half].sum(axis=1), 99.85)) * 1.20, 4.0)
        k_pref_strict = max(float(np.percentile(vpref[:half].sum(axis=1), 99.85)) * 1.18, 5.5)
        hold_strict = np.maximum(
            np.maximum(
                (vb128[half + S :].sum(axis=1) > k_128_strict).astype(int),
                (vpref[half + S :].sum(axis=1) > k_pref_strict).astype(int),
            ),
            edge_hold,
        )
        test_strict = np.maximum(
            np.maximum(
                (tb128.sum(axis=1) > k_128_strict).astype(int),
                (tpref.sum(axis=1) > k_pref_strict).astype(int),
            ),
            edge_test,
        )

        return CircuitBreakerStreamDecision(
            valid_mask=tm,
            cal_switch_counts=tot_va[:half],
            hold_switch_counts=tot_va[half + S :],
            test_switch_counts=tot_te,
            hold_pred_strict=hold_strict,
            test_pred_strict=test_strict,
            hold_pred_balanced=hold_bal,
            test_pred_balanced=test_bal,
            k_gate_strict=float(k_pref_strict),
            k_gate_balanced=float(k_pref_bal),
        )


@torch.no_grad()
def extract_multihorizon_4subblocks(
    model: nn.Module,
    windows: np.ndarray,
    context_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (raw4, white4) of shape (N_win, 4, D) for SpeculativeJEPAModel or SolitonReynoldsJEPAModel."""
    model.eval()
    raw4_list, white4_list = [], []
    with torch.inference_mode():
        for i in range(0, len(windows), 128):
            ch = torch.from_numpy(np.ascontiguousarray(windows[i : i + 128])).float().to(device)
            ctx, tgt = ch[:, :context_len], ch[:, context_len:]
            out = model.forward(ctx, tgt)
            if len(out) == 3:
                # SpeculativeJEPAModel: (z_ctx, z_targets [B,3,D], z_drafts [B,3,D])
                _, z_targets, z_drafts = out
                diffs3 = z_targets - z_drafts
                bsz, n_h, dim = diffs3.shape
                e_white3 = whiten_residuals(
                    diffs3.reshape(bsz, n_h * dim) - model.residual_mean, model.precision_matrix
                ).reshape(bsz, n_h, dim)
                raw4 = torch.stack(
                    [diffs3[:, 0], diffs3[:, 1], 0.5 * (diffs3[:, 1] + diffs3[:, 2]), diffs3[:, 2]], dim=1
                )
                white4 = torch.stack(
                    [e_white3[:, 0], e_white3[:, 1], 0.5 * (e_white3[:, 1] + e_white3[:, 2]), e_white3[:, 2]], dim=1
                )
            else:
                # SolitonReynoldsJEPAModel: (z_ctx, z_tgt, z_pred, z_sub_pred [B,4,D], z_sub_tgt [B,4,D])
                _, _, _, z_sub_pred, z_sub_tgt = out
                raw4 = z_sub_tgt - z_sub_pred
                bsz, n_m, dim = raw4.shape
                white4 = whiten_residuals(
                    raw4.reshape(bsz, n_m * dim) - model.residual_mean, model.precision_matrix
                ).reshape(bsz, n_m, dim)

            raw4_list.append(raw4.cpu().numpy().astype(np.float32))
            white4_list.append(white4.cpu().numpy().astype(np.float32))

    return np.concatenate(raw4_list, axis=0), np.concatenate(white4_list, axis=0)
