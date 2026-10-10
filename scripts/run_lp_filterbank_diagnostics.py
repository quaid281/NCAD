#!/usr/bin/env python3
"""Littlewood-Paley Dyadic Filterbank Diagnostics (Reviewer Major Concern 7).

The manuscript describes the frequency front-end as "1D depthwise convolutions
with kernel lengths {11, 7, 5, 3}" implementing the Littlewood-Paley partition.
This script verifies the actual implementation in
`src/models/geometric_layers.py::LittlewoodPaleyDyadicBlock`:

1. The dyadic partition of unity is constructed in the FOURIER domain
   (`compute_dyadic_filters`), NOT by the time-domain convolutions.
2. The shell convolutions are full (groups=1) convolutions with kernel sizes
   (15, 9, 5, 3) for num_shells=4, applied per shell AFTER the exact
   Fourier-domain split; they are feature processors, not the partition itself.

Outputs:
- reports/lp_filterbank_frequency_responses.csv
- reports/lp_filterbank_diagnostics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.geometric_layers import LittlewoodPaleyDyadicBlock


def conv_frequency_response(conv: torch.nn.Conv1d, n_fft: int = 1024) -> np.ndarray:
    """Mean magnitude response |H(omega)| of a Conv1d kernel across all
    (out_channel, in_channel) pairs, evaluated on the nonnegative rfft grid."""
    weight = conv.weight.detach().cpu().double()  # (C_out, C_in, K)
    spectra = torch.fft.rfft(weight, n=n_fft, dim=-1)  # (C_out, C_in, F)
    mag = spectra.abs().mean(dim=(0, 1)).numpy()
    return mag


def ideal_dyadic_bands(omega: np.ndarray, J: int) -> np.ndarray:
    """Ideal (rectangular) dyadic partition of [0,1] into J octave bands."""
    edges = np.concatenate([[0.0], 2.0 ** np.arange(-(J - 1), 0), [1.0]])
    bands = np.zeros((J, len(omega)))
    for j in range(J):
        bands[j] = ((omega >= edges[j]) & (omega <= edges[j + 1])).astype(float)
    bands[0, 0] = 1.0
    return bands


def band_half_power(omega: np.ndarray, psi: np.ndarray) -> tuple[float, float, float]:
    """Peak frequency and effective half-power band edges of a shell response."""
    peak_idx = int(np.argmax(psi))
    peak_f = float(omega[peak_idx])
    half = 0.5 * float(psi[peak_idx])
    above = omega[psi >= half]
    lo = float(above.min()) if len(above) else peak_f
    hi = float(above.max()) if len(above) else peak_f
    return peak_f, lo, hi


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    channels = 48
    num_shells = 4
    T = 256  # manuscript context length C = 256
    block = LittlewoodPaleyDyadicBlock(channels=channels, num_shells=num_shells)
    block.eval()

    # ------------------------------------------------------------------
    # 1. Fourier-domain partition of unity: exactness of sum_j Psi_j(omega)
    # ------------------------------------------------------------------
    F_bins = T // 2 + 1
    psi = block.compute_dyadic_filters(F_bins, device=torch.device("cpu"))  # (J,1,F)
    psi_np = psi[:, 0, :].numpy()
    omega = np.linspace(0.0, 1.0, F_bins)

    pou_sum = psi_np.sum(axis=0)
    pou_max_err = float(np.max(np.abs(pou_sum - 1.0)))

    # ------------------------------------------------------------------
    # 2. Reconstruction fidelity: x == sum_j irfft(Psi_j * rfft(x))
    # ------------------------------------------------------------------
    x = torch.randn(8, channels, T, dtype=torch.float64)
    X = torch.fft.rfft(x, dim=-1)
    x_shells = torch.fft.irfft(X.unsqueeze(0) * psi.unsqueeze(1).double(), n=T, dim=-1)
    recon = x_shells.sum(dim=0)
    recon_rel_err = float(
        (recon - x).abs().max() / x.abs().max()
    )
    shell_energy = x_shells.pow(2).sum(dim=(1, 2, 3)).numpy()  # (J,)
    total_energy = float(x.pow(2).sum())
    parseval_err = float(abs(shell_energy.sum() - total_energy) / total_energy)

    # ------------------------------------------------------------------
    # 3. Time-domain convolution responses (the learned per-shell processors)
    # ------------------------------------------------------------------
    conv_responses = []
    for j, shell in enumerate(block.shell_convs):
        conv = shell[0]
        resp = conv_frequency_response(conv)
        # Resample conv response (n_fft=1024 grid) onto the T=256 omega grid
        conv_omega = np.linspace(0.0, 1.0, resp.shape[0])
        conv_responses.append(np.interp(omega, conv_omega, resp))

    conv_responses = np.stack(conv_responses)

    # ------------------------------------------------------------------
    # 4. Ideal dyadic comparison + shell descriptors
    # ------------------------------------------------------------------
    ideal = ideal_dyadic_bands(omega, num_shells)

    diagnostics = {
        "num_shells": num_shells,
        "kernel_sizes_configured": str(list(block.kernel_sizes)),
        "paper_claimed_kernel_sizes": "{11, 7, 5, 3}",
        "conv_groups": int(block.shell_convs[0][0].groups),
        "conv_is_depthwise": bool(block.shell_convs[0][0].groups == channels),
        "partition_of_unity_max_err": pou_max_err,
        "reconstruction_max_rel_err": recon_rel_err,
        "parseval_energy_rel_err": parseval_err,
        "causal_left_padding": str([k - 1 for k in block.kernel_sizes]),
    }

    for j in range(num_shells):
        peak_f, lo, hi = band_half_power(omega, psi_np[j])
        diagnostics[f"shell_{j}_peak_freq"] = peak_f
        diagnostics[f"shell_{j}_halfpower_lo"] = lo
        diagnostics[f"shell_{j}_halfpower_hi"] = hi
        # Fraction of shell mass that lies inside the ideal octave band
        in_band = float(
            np.trapezoid(psi_np[j] * ideal[j], omega)
            / max(np.trapezoid(psi_np[j], omega), 1e-12)
        )
        diagnostics[f"shell_{j}_mass_in_ideal_band"] = in_band
        diagnostics[f"shell_{j}_signal_energy_share"] = float(
            shell_energy[j] / shell_energy.sum()
        )

    # ------------------------------------------------------------------
    # 5. Per-frequency response table
    # ------------------------------------------------------------------
    resp_df = pd.DataFrame({"omega": omega})
    for j in range(num_shells):
        resp_df[f"psi_shell_{j}"] = psi_np[j]
        resp_df[f"ideal_band_{j}"] = ideal[j]
        resp_df[f"conv_response_shell_{j}"] = conv_responses[j]

    out_resp = ROOT / "reports" / "lp_filterbank_frequency_responses.csv"
    out_diag = ROOT / "reports" / "lp_filterbank_diagnostics.csv"
    resp_df.to_csv(out_resp, index=False)
    pd.DataFrame([diagnostics]).to_csv(out_diag, index=False)

    print("=== Littlewood-Paley Filterbank Diagnostics ===")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_resp}")
    print(f"Wrote {out_diag}")


if __name__ == "__main__":
    main()
