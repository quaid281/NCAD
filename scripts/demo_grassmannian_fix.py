"""Prototype demo: Grassmannian codebook load-balancing fix.

Compares three codebook configurations on a spectrally heterogeneous probe mix
(frequency-scaled telemetry, scales 0.25x - 4x):

  A) Stock QR-random init (baseline)
  B) Spectral-band init: init_spectral_band_frames() seeds each regime onto an
     interleaved slice of the context covariance eigenbasis
  C) Spectral-band init + brief optimization of the composite Delsarte +
     Switch-style load-balancing aux loss

For each variant we report overall winner histogram, per-scale routing,
winner margins, and soft-routing entropy at the configured temperature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.models import HybridTCNEncoder
from src.models.jepa.potential_flow_jepa import HarmonicGrassmannianCodebook

try:
    from scripts.diagnose_grassmannian import make_probe, energies_for
except ImportError:
    from diagnose_grassmannian import make_probe, energies_for

SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]


def eval_codebook(cb, frames, Z_mix, Z_per_scale, device) -> Dict:
    cb.eval()
    E = energies_for(cb, Z_mix, frames)
    winners = E.argmax(-1)
    hist = winners.bincount(minlength=cb.n_regimes).tolist()
    E_sorted, _ = E.sort(-1, descending=True)
    margin = (E_sorted[:, 0] - E_sorted[:, 1]).mean().item()
    probs = F.softmax(E / cb.temperature, dim=-1)
    H = -(probs * torch.log(probs + 1e-12)).sum(-1)
    usage = torch.tensor(hist, dtype=torch.float) / max(sum(hist), 1)
    usage_H = float((-(usage * torch.log(usage + 1e-12)).sum() / np.log(cb.n_regimes)).item())

    per_scale = {}
    for s, Z in Z_per_scale.items():
        E_s = energies_for(cb, Z, frames)
        per_scale[f"x{s}"] = E_s.argmax(-1).bincount(minlength=cb.n_regimes).tolist()

    return {
        "winner_hist": hist,
        "usage_entropy_norm": round(usage_H, 3),
        "winner_margin": round(margin, 4),
        "soft_entropy": round(float(H.mean().item()), 4),
        "eff_regimes": round(float(torch.exp(H).mean().item()), 3),
        "per_scale_winners": per_scale,
    }


def train_load_balance(cb, Z, steps: int = 500, lr: float = 3e-2, train_tau: float = 5.0) -> List[float]:
    """Optimize raw_frames on the composite ortho + load-balance loss.

    Must train at a high softmax temperature (~5.0): at the eval temperature
    tau=0.1 routing is already saturated (one-hot), leaving near-zero gradient
    through the mean soft mass P_k — the aux loss is only useful during the
    high-temperature exploratory phase.
    """
    eval_tau = cb.temperature
    cb.temperature = train_tau
    cb.train()
    opt = torch.optim.Adam([cb.raw_frames], lr=lr)
    losses = []
    meanP_hist = []
    for i in range(steps):
        opt.zero_grad()
        _, probs, loss = cb(Z, hard=False)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if i % 100 == 0:
            meanP_hist.append([round(float(x), 3) for x in probs.mean(0).tolist()])
    cb.temperature = eval_tau
    cb.eval()
    print(f"    [train] meanP trajectory: {meanP_hist}")
    return losses


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Grassmannian load-balance prototype on {device}\n")

    # Shared frozen encoder -> isolate codebook behavior
    encoder = HybridTCNEncoder(input_dim=9, latent_dim=32, filters=48, tcn_layers=3).to(device).eval()

    Z_per_scale: Dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for s in SCALES:
            probe = make_probe(batch_size=64, context_len=256, channels=9, freq_scale=s, seed=7)
            Z_per_scale[s] = encoder(probe.to(device))
    Z_mix = torch.cat(list(Z_per_scale.values()), dim=0)  # (320, 32)
    print(f"Probe mix: {len(Z_mix)} samples across {len(SCALES)} frequency scales")

    variants = {}

    # ---- A: stock random init ----
    torch.manual_seed(42)
    cb_A = HarmonicGrassmannianCodebook(latent_dim=32, n_regimes=4, subspace_dim=8, temperature=0.1).to(device)
    variants["A_random_init"] = (cb_A, cb_A.get_orthonormal_frames().detach())

    # ---- B: spectral-band init ----
    torch.manual_seed(42)
    cb_B = HarmonicGrassmannianCodebook(latent_dim=32, n_regimes=4, subspace_dim=8, temperature=0.1).to(device)
    cb_B.init_spectral_band_frames(Z_mix)
    variants["B_spectral_band_init"] = (cb_B, cb_B.get_orthonormal_frames().detach())

    # ---- C: spectral-band init + load-balance training ----
    torch.manual_seed(42)
    cb_C = HarmonicGrassmannianCodebook(
        latent_dim=32, n_regimes=4, subspace_dim=8, temperature=0.1, load_balance_weight=0.5
    ).to(device)
    cb_C.init_spectral_band_frames(Z_mix)
    losses = train_load_balance(cb_C, Z_mix, steps=500, train_tau=5.0)
    variants["C_band_init+lb_trained"] = (cb_C, cb_C.get_orthonormal_frames().detach())

    print("\n" + "=" * 96)
    for name, (cb, frames) in variants.items():
        res = eval_codebook(cb, frames, Z_mix, Z_per_scale, device)
        print(f"\n--- {name} ---")
        print(f"  overall winners: {res['winner_hist']}   usage_H(norm)={res['usage_entropy_norm']}")
        print(f"  margin={res['winner_margin']}  softH={res['soft_entropy']}  eff#regimes={res['eff_regimes']}")
        for s in SCALES:
            print(f"    freq x{s:<5}: {res['per_scale_winners'][f'x{s}']}")

    print(f"\n[C] composite loss: {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    # ---- Soft-eval routing on the lb-trained codebook ----
    # hard winners collapse to one regime, but with soft routing at eval
    # (hard_eval=False) all regimes contribute through the balanced mixture.
    cb_C.hard_eval = False
    cb_C.eval()
    print("\n--- C under SOFT eval routing (hard_eval=False): per-scale mean probs ---")
    with torch.no_grad():
        for s in SCALES:
            _, probs, _ = cb_C(Z_per_scale[s], hard=False)
            print(f"    freq x{s:<5}: meanP={[round(float(x), 3) for x in probs.mean(0).tolist()]}")


if __name__ == "__main__":
    main()
