"""Position+Drift (P+D) Single-Quorum Circuit-Breaker Panel for Laminar-JEPA-Lite.

Replaces the 224-dial + 32-relay + 3-quorum panel with `(M + 1) * D` nonnegative
dials (`M * D` sub-horizon position magnitudes + `D` Reynolds horizon-mean drift
magnitudes = 128 dials at `M = 3, D = 32`, or 160 dials at `M = 4, D = 32`)
governed by a single nominal empirical quorum and zero onset relays.
"""

from __future__ import annotations

import numpy as np

from src.scoring.circuit_breaker_panel import (
    CircuitBreakerStreamDecision,
    aggregate_vector_stream,
)


def build_position_drift_dials(ew_signed: np.ndarray) -> np.ndarray:
    """Expand `(N_win, M, D)` signed residuals into `(N_win, (M + 1) * D)` P+D dials."""
    if ew_signed.ndim != 3:
        raise ValueError(f"Expected 3D residual array (N_win, M, D), got shape {ew_signed.shape}.")
    n_win, n_horizons, dim = ew_signed.shape
    if n_horizons < 2:
        raise ValueError(f"Expected M >= 2 horizons along axis=1, got {n_horizons}.")
    d_pos = np.abs(ew_signed).reshape(n_win, n_horizons * dim)
    d_drift = np.abs(ew_signed.mean(axis=1))
    return np.concatenate([d_pos, d_drift], axis=1)


def _prefix_spans_for_horizons(suspect_size: int, num_horizons: int) -> list[int]:
    if num_horizons == 3:
        return [max(1, suspect_size // 4), max(2, suspect_size // 2), suspect_size]
    return [
        suspect_size if m == num_horizons - 1 else max(1, ((m + 1) * suspect_size) // num_horizons)
        for m in range(num_horizons)
    ]


def aggregate_prefix_pd_stream(
    bits_pd: np.ndarray,
    n_points: int,
    context_size: int,
    suspect_size: int,
    num_horizons: int,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project `(N_win, (M + 1) * D)` P+D switch states onto the stream timeline."""
    if mapping == "trailing":
        return aggregate_vector_stream(bits_pd, n_points, context_size, suspect_size, "trailing")
    if mapping != "smear":
        raise ValueError(f"Unsupported mapping '{mapping}'. Expected 'trailing' or 'smear'.")

    n_win, total_dim = bits_pd.shape
    dim = total_dim // (num_horizons + 1)
    diff_arr = np.zeros((n_points + 1, total_dim), dtype=np.float64)
    diff_cnt = np.zeros((n_points + 1, total_dim), dtype=np.float64)
    starts = np.arange(n_win) + context_size

    prefix_ends = _prefix_spans_for_horizons(suspect_size, num_horizons)
    spans = [(m * dim, (m + 1) * dim, prefix_ends[m]) for m in range(num_horizons)]
    spans.append((num_horizons * dim, (num_horizons + 1) * dim, suspect_size))

    for c0, c1, span_len in spans:
        e_idx = np.minimum(starts + span_len, n_points)
        valid = (starts < n_points) & (e_idx > starts)
        np.add.at(diff_arr[:, c0:c1], starts[valid], bits_pd[valid, c0:c1])
        np.add.at(diff_arr[:, c0:c1], e_idx[valid], -bits_pd[valid, c0:c1])
        np.add.at(diff_cnt[:, c0:c1], starts[valid], 1.0)
        np.add.at(diff_cnt[:, c0:c1], e_idx[valid], -1.0)

    out = np.cumsum(diff_arr[:-1], axis=0)
    cnt = np.cumsum(diff_cnt[:-1], axis=0)
    mask = cnt > 0
    out[mask] /= cnt[mask]
    valid_t = cnt[:, num_horizons * dim] > 0
    return out.astype(np.float32), valid_t


class PositionDriftLitePanel:
    """Single-quorum `(M + 1) * D` Position+Drift circuit-breaker panel (`128` dials at `M=3`)."""

    def __init__(
        self,
        context_size: int = 256,
        suspect_size: int = 64,
        use_hijack_guard: bool = True,
        use_prefix_smear: bool = True,
    ):
        self.context_size = context_size
        self.suspect_size = suspect_size
        self.use_hijack_guard = use_hijack_guard
        self.use_prefix_smear = use_prefix_smear

    def calibrate_and_trip_windows(
        self, va_ew: np.ndarray, te_ew: np.ndarray, n_cal_w: int
    ) -> dict[str, np.ndarray | int]:
        """Calibrate per-dial 98.5th-percentile switches on `va_ew[:n_cal_w]`."""
        _, num_horizons, dim = va_ew.shape
        va_pd = build_position_drift_dials(va_ew)
        te_pd = build_position_drift_dials(te_ew)

        med = np.median(va_pd[:n_cal_w], axis=0, keepdims=True)
        mad = 1.4826 * np.median(np.abs(va_pd[:n_cal_w] - med), axis=0, keepdims=True) + 1e-4
        z_va = np.maximum(0.0, (va_pd - med) / mad)
        z_te = np.maximum(0.0, (te_pd - med) / mad)

        if self.use_hijack_guard:
            q1 = max(1, n_cal_w // 2)
            drift = np.maximum(
                1.0,
                (np.percentile(z_va[q1:n_cal_w], 98.0, axis=0, keepdims=True) + 0.5)
                / (np.percentile(z_va[:q1], 98.0, axis=0, keepdims=True) + 0.5),
            )
            s_cal = med + mad
            s_ref = (
                np.concatenate(
                    [
                        np.repeat(np.median(s_cal[:, : num_horizons * dim], keepdims=True), num_horizons * dim, axis=1),
                        np.repeat(np.median(s_cal[:, num_horizons * dim :], keepdims=True), dim, axis=1),
                    ],
                    axis=1,
                )
                + 1e-4
            )
            scale = np.maximum(1.0, s_cal / (1.5 * s_ref))
            guard = np.where(drift > 1.35, drift, 1.0) * np.where(scale > 1.25, scale, 1.0)
        else:
            guard = 1.0

        tau = (np.percentile(z_va[:n_cal_w], 98.5, axis=0, keepdims=True) + 1e-4) * guard
        return {
            "bits_pd_va": (z_va > tau).astype(np.float32),
            "bits_pd_te": (z_te > tau).astype(np.float32),
            "num_horizons": int(num_horizons),
        }

    def evaluate_stream_decisions(
        self,
        panel_bits: dict[str, np.ndarray | int],
        len_va: int,
        len_test: int,
        mapping: str,
    ) -> CircuitBreakerStreamDecision:
        """Project binary P+D switches onto the timeline and apply a single nominal quorum."""
        c_len, s_len = self.context_size, self.suspect_size
        bits_va = np.asarray(panel_bits["bits_pd_va"], dtype=np.float32)
        bits_te = np.asarray(panel_bits["bits_pd_te"], dtype=np.float32)
        num_horizons = int(panel_bits["num_horizons"])

        if mapping == "smear" and self.use_prefix_smear:
            vp, vm = aggregate_prefix_pd_stream(bits_va, len_va, c_len, s_len, num_horizons, mapping)
            tp, tm = aggregate_prefix_pd_stream(bits_te, len_test, c_len, s_len, num_horizons, mapping)
        else:
            vp, vm = aggregate_vector_stream(bits_va, len_va, c_len, s_len, mapping)
            tp, tm = aggregate_vector_stream(bits_te, len_test, c_len, s_len, mapping)

        v_cnt = vp[vm].sum(axis=1)
        t_cnt = tp[tm].sum(axis=1)
        half = len(v_cnt) // 2
        n_dials = float(bits_va.shape[1])

        floor_bal = max(2.0, 3.5 * (n_dials / 128.0))
        floor_strict = max(2.5, 4.0 * (n_dials / 128.0))
        k_bal = max(float(np.percentile(v_cnt[:half], 99.5)) * 1.12, floor_bal)
        k_strict = max(float(np.percentile(v_cnt[:half], 99.85)) * 1.18, floor_strict)

        hold_cnt = v_cnt[half + s_len :]
        return CircuitBreakerStreamDecision(
            valid_mask=tm,
            cal_switch_counts=v_cnt[:half],
            hold_switch_counts=hold_cnt,
            test_switch_counts=t_cnt,
            hold_pred_strict=(hold_cnt > k_strict).astype(int),
            test_pred_strict=(t_cnt > k_strict).astype(int),
            hold_pred_balanced=(hold_cnt > k_bal).astype(int),
            test_pred_balanced=(t_cnt > k_bal).astype(int),
            k_gate_strict=k_strict,
            k_gate_balanced=k_bal,
        )


__all__ = [
    "PositionDriftLitePanel",
    "aggregate_prefix_pd_stream",
    "build_position_drift_dials",
]
