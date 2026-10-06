"""Grassmannian Codebook Saturation Diagnosis for PotentialFlowJEPA.

Dissects why the HarmonicGrassmannianCodebook routes all context embeddings to
a subset of regimes:

1. Routing-mode artifact: eval-mode forward takes hard argmax one-hot by
   construction (entropy is *defined* as 0). We separately probe the soft path
   (softmax(energy / tau)) across temperatures to measure true confidence.
2. Frame quality: per-regime captured energy distributions, winner margins,
   pairwise dominance matrix P(E_i > E_j), and principal angles between each
   frame U_k and the top-d PCs of the z_ctx covariance (explains which frames
   are aligned with the data manifold at init).
3. Downstream sensitivity: counterfactual v_pred spread when regime routing is
   forced to each k — does the dead/alive regime split matter to the flow?
4. Input sensitivity: winner histograms under frequency-rescaled probe
   telemetry — does routing respond to different dynamics at all?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.models import PotentialFlowJEPAModel, HybridTCNEncoder

try:
    from scripts.trace_deep_activations import build_potential_flow
    from scripts.trace_model_activations import generate_synthetic_telemetry
except ImportError:
    from trace_deep_activations import build_potential_flow
    from trace_model_activations import generate_synthetic_telemetry


def make_probe(
    batch_size: int = 64,
    context_len: int = 256,
    channels: int = 9,
    freq_scale: float = 1.0,
    seed: int = 7,
) -> torch.Tensor:
    """Nominal-dynamics probe context with rescaled spectral content."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10 * np.pi, context_len)
    out = []
    for b in range(batch_size):
        chans = []
        for c in range(channels):
            freq = (0.5 + 0.1 * c) * freq_scale
            phase = (c * np.pi) / channels
            chans.append(np.sin(freq * t + phase) + 0.3 * np.cos(2 * freq * t)
                         + rng.normal(0, 0.05, context_len))
        out.append(np.stack(chans, axis=-1))
    return torch.tensor(np.array(out, dtype=np.float32))


def energies_for(cb, z: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """(B, K) captured subspace energies ||U_k^T z||^2."""
    proj = torch.einsum("bd,kdj->bkj", z, frames)
    return proj.pow(2).sum(-1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Grassmannian codebook diagnosis on {device}")

    B = 256
    ctx_nom, tgt_nom, _, _ = generate_synthetic_telemetry(
        batch_size=B, context_len=256, target_len=64, channels=9, seed=42
    )
    model = build_potential_flow(ctx_nom.size(-1), device)
    cb = model.grassmannian_codebook
    assert cb is not None

    with torch.no_grad():
        z_ctx = model.context_encoder(ctx_nom.to(device))
        z_tgt = model.target_encoder(tgt_nom.to(device))
    frames = cb.get_orthonormal_frames().detach()  # (K, D, d)
    K, D, d = frames.shape
    with torch.no_grad():
        E = energies_for(cb, z_ctx, frames)  # (B, K)

    report: Dict[str, Any] = {"n_regimes": K, "latent_dim": D, "subspace_dim": d,
                              "temperature": cb.temperature}

    # ------------------------------------------------------------------ #
    # 1. Energy statistics, winner margins, dominance structure
    # ------------------------------------------------------------------ #
    winners = E.argmax(-1)
    hist = winners.bincount(minlength=K).tolist()
    E_sorted, _ = E.sort(-1, descending=True)
    margin = E_sorted[:, 0] - E_sorted[:, 1]
    dominance = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            if i != j:
                dominance[i, j] = (E[:, i] > E[:, j]).float().mean()

    report["energy_stats"] = {
        "per_regime_mean": [round(float(E[:, k].mean()), 4) for k in range(K)],
        "per_regime_std": [round(float(E[:, k].std()), 4) for k in range(K)],
        "winner_hist": hist,
        "margin_e1_minus_e2_mean": float(margin.mean().item()),
        "margin_over_energy_mean": float((margin / (E_sorted[:, 0] + 1e-8)).mean().item()),
        "dominance_P(Ei>Ej)": [[round(float(dominance[i, j]), 3) for j in range(K)] for i in range(K)],
    }

    print("\n[1] Subspace energy capture (nominal batch)")
    for k in range(K):
        print(f"    regime {k}: E={E[:, k].mean():.3f}+/-{E[:, k].std():.3f}  wins={hist[k]}")
    print(f"    winner margin (E1-E2): {margin.mean():.3f} "
          f"({(margin / (E_sorted[:, 0] + 1e-8)).mean() * 100:.1f}% of top energy)")
    print("    dominance matrix P(E_i > E_j):")
    print("          " + "        ".join(f"j={j}" for j in range(K)))
    for i in range(K):
        print(f"    i={i}  " + "        ".join(f"{dominance[i, j]:.3f}" for j in range(K)))

    # ------------------------------------------------------------------ #
    # 2. Soft vs hard routing — temperature sweep
    # ------------------------------------------------------------------ #
    print("\n[2] Softmax confidence sweep (soft routing path)")
    report["softmax_sweep"] = {}
    for tau in [0.01, 0.1, 0.5, 1.0, 5.0]:
        probs = F.softmax(E / tau, dim=-1)
        H = -(probs * torch.log(probs + 1e-12)).sum(-1)
        report["softmax_sweep"][str(tau)] = {
            "mean_entropy": float(H.mean().item()),
            "eff_regimes_expH": float(torch.exp(H).mean().item()),
            "mean_max_prob": float(probs.max(-1).values.mean().item()),
        }
        print(f"    tau={tau:<5}: H={H.mean():.4f}  eff#regimes={torch.exp(H).mean():.3f}  "
              f"maxp={probs.max(-1).values.mean():.4f}")

    # logit gaps in energy units -> implied confidence at trained tau
    logit_gap = margin / cb.temperature
    report["energy_margin_over_tau"] = float(logit_gap.mean().item())
    print(f"    implied logit gap at tau={cb.temperature}: {logit_gap.mean():.1f} "
          f"(e^gap odds = {float(torch.exp(logit_gap.mean()).item()):.3g}:1)")

    # ------------------------------------------------------------------ #
    # 3. Principal angles: frame alignment with data manifold
    # ------------------------------------------------------------------ #
    print("\n[3] Frame alignment with top-8 PCs of z_ctx covariance")
    Zc = z_ctx.cpu() - z_ctx.cpu().mean(0)
    cov = Zc.T @ Zc / (Zc.size(0) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    V = evecs[:, -d:].cpu()  # top-d PCs, (D, d)
    frames_cpu = frames.cpu()
    align = []
    for k in range(K):
        M = frames_cpu[k].T @ V  # (d, d) singular vals = cos principal angles
        sv = torch.linalg.svdvals(M)
        align.append({
            "regime": k,
            "cos2_principal_angles": [round(float(s * s), 4) for s in sv.flip(0)],
            "captured_fraction_of_topd": float((sv ** 2).mean().item()),
        })
        print(f"    regime {k}: mean cos^2(theta)={float((sv**2).mean()):.4f}  "
              f"angles={[round(float(s*s),3) for s in sv.flip(0)]}")
    report["frame_pc_alignment"] = align

    # cross-frame overlap (Delsarte term decomposition)
    overlaps = []
    for i in range(K):
        for j in range(i + 1, K):
            overlaps.append(float((frames_cpu[i].T @ frames_cpu[j]).pow(2).sum().item()))
    report["delsarte_pairwise_overlaps"] = overlaps
    print(f"    Delsarte pairwise ||U_i^T U_j||_F^2: {[round(o,3) for o in overlaps]} "
          f"(random-init expectation d^2/D={d*d/D:.2f})")

    # ------------------------------------------------------------------ #
    # 4. Counterfactual regime sensitivity of the velocity field
    # ------------------------------------------------------------------ #
    print("\n[4] Counterfactual v_pred spread under forced regimes")
    t_mid = torch.full((B,), 0.5, device=device)
    z_mid = 0.5 * z_tgt
    with torch.no_grad():
        v_forced = []
        for k in range(K):
            p_k = (z_ctx @ frames[k]) @ frames[k].T
            v_k = model.flow_predictor(z_mid, t_mid, z_ctx, p_regime=p_k, create_graph=False)
            v_forced.append(v_k)
        V_stack = torch.stack(v_forced, dim=1)  # (B, K, D)
        v_mean = V_stack.mean(1, keepdim=True)
        spread = (V_stack - v_mean).norm(dim=-1).mean()
        base_norm = V_stack.norm(dim=-1).mean()
    report["regime_sensitivity"] = {
        "v_norm_mean": float(base_norm.item()),
        "forced_regime_spread": float(spread.item()),
        "spread_over_norm": float((spread / (base_norm + 1e-8)).item()),
    }
    print(f"    ||v|| mean={base_norm:.4f}  forced-regime spread={spread:.4f} "
          f"({float(spread / (base_norm + 1e-8)) * 100:.1f}% of magnitude)")

    # ------------------------------------------------------------------ #
    # 5. Input sensitivity: does routing respond to different dynamics?
    # ------------------------------------------------------------------ #
    print("\n[5] Winner histograms under frequency-rescaled probe contexts")
    report["probe_histograms"] = {}
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        probe = make_probe(batch_size=64, context_len=256, channels=z_ctx.size(-1)
                           if z_ctx.ndim > 2 else 9, freq_scale=scale).to(device)
        with torch.no_grad():
            z_probe = model.context_encoder(probe)
        E_p = energies_for(cb, z_probe, frames)
        h = E_p.argmax(-1).bincount(minlength=K).tolist()
        report["probe_histograms"][f"freq_scale_{scale}"] = h
        print(f"    freq x{scale:<5}: winners={h}  meanE={[round(float(E_p[:,k].mean()),2) for k in range(K)]}")

    out_dir = Path("reports/latent_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grassmannian_diagnosis.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
