"""Latent Space Diagnosis for the Physics-Informed JEPA Triad.

Extracts latent embeddings (z_ctx, z_tgt, z_pred / v_pred) at batch scale and
diagnoses representation-space health and anomaly separability:

- Covariance spectrum per latent set: eigenvalues, effective rank,
  participation ratio, top-PC share, min eigenvalue (collapse detection).
- Isotropy / uniformity: pairwise cosine statistics, log-uniformity energy
  (Wang & Isola), and positive-pair alignment (z_ctx vs z_tgt of same sample).
- Nominal vs. anomalous separation: centroid displacement in Mahalanobis
  units, Mahalanobis-distance AUROC, kNN-distance AUROC, model discrepancy
  AUROC, and per-dimension shift ranking.
- Predictive residual geometry: residual norm distributions, per-dim squared
  contributions, and cosine alignment of the mean anomalous residual with the
  top principal components of the nominal manifold.
- Model-specific extras: Grassmannian regime usage (PotentialFlow),
  Reynolds stress cone eigen-spectrum (ReynoldsStress).

Note: the synthetic fault is injected only into the target horizon, so
z_ctx is identical for nominal and anomalous batches; all separation metrics
are computed on z_tgt and residuals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.models import (
    HybridTCNEncoder,
    OperatorEntropyJEPAModel,
    PotentialFlowJEPAModel,
    ReynoldsStressJEPAModel,
)

try:
    from scripts.trace_deep_activations import (
        build_operator_entropy,
        build_potential_flow,
        build_reynolds_stress,
    )
    from scripts.trace_model_activations import generate_synthetic_telemetry
except ImportError:
    from trace_deep_activations import (
        build_operator_entropy,
        build_potential_flow,
        build_reynolds_stress,
    )
    from trace_model_activations import generate_synthetic_telemetry


# --------------------------------------------------------------------------- #
# Latent extraction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_latents(
    model: nn.Module,
    model_key: str,
    ctx: torch.Tensor,
    tgt: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Return {z_ctx, z_tgt, z_pred, residual} on CPU."""
    ctx, tgt = ctx.to(device), tgt.to(device)
    if model_key == "operator_entropy":
        z_pred, z_tgt, _ = model(ctx, tgt)
        z_ctx = model._apply_filters(model.context_encoder(ctx))
        residual = z_tgt - z_pred
    elif model_key == "reynolds_stress":
        z_pred, z_tgt, _ = model(ctx, tgt)
        z_ctx = model._extract_context(ctx)
        residual = z_tgt - z_pred
    elif model_key == "potential_flow":
        B = ctx.size(0)
        t_mid = torch.full((B,), 0.5, device=device, dtype=ctx.dtype)
        z_zero = torch.zeros(B, model.latent_dim, device=device, dtype=ctx.dtype)
        z_ctx, z_tgt, v_pred, _, _ = model(ctx, tgt, t=t_mid, z_noise=z_zero)
        # Matches compute_predictive_discrepancy: score on v_pred - z_tgt
        z_pred = v_pred
        residual = z_tgt - v_pred
    else:
        raise ValueError(model_key)
    return {
        "z_ctx": z_ctx.detach().cpu().float(),
        "z_tgt": z_tgt.detach().cpu().float(),
        "z_pred": z_pred.detach().cpu().float(),
        "residual": residual.detach().cpu().float(),
    }


# --------------------------------------------------------------------------- #
# Geometry metrics
# --------------------------------------------------------------------------- #

def _cov_eigs(Z: torch.Tensor) -> torch.Tensor:
    Zc = Z - Z.mean(0, keepdim=True)
    cov = (Zc.T @ Zc) / max(Z.size(0) - 1, 1)
    return torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)


def spectrum_stats(Z: torch.Tensor) -> Dict[str, float]:
    eigs = _cov_eigs(Z)
    total = eigs.sum().item()
    p = eigs / (eigs.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    part_ratio = float((eigs.sum() ** 2 / (eigs.pow(2).sum() + 1e-12)).item())
    var = Z.var(dim=0, unbiased=False)
    return {
        "n_samples": int(Z.size(0)),
        "latent_dim": int(Z.size(1)),
        "mean_norm": float(Z.norm(dim=-1).mean().item()),
        "std_norm": float(Z.norm(dim=-1).std().item()),
        "eff_rank": eff_rank,
        "participation_ratio": part_ratio,
        "top1_var_share": float(eigs[0].item() / (total + 1e-12)),
        "top5_var_share": float(eigs[:5].sum().item() / (total + 1e-12)),
        "min_eig": float(eigs[-1].item()),
        "dead_dims_var<1e-6": int((var < 1e-6).sum().item()),
        "eig_spectrum": [round(float(e), 6) for e in eigs[:10]],
    }


def isotropy_stats(Z: torch.Tensor) -> Dict[str, float]:
    Zn = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
    G = Zn @ Zn.T
    mask = ~torch.eye(Z.size(0), dtype=torch.bool)
    cos = G[mask]
    # Uniformity: log mean exp(-2 ||z_i - z_j||^2) = log mean exp(-4 + 4 cos)
    uniformity = float(torch.log(torch.exp(-4.0 + 4.0 * G[mask]).mean()).item())
    return {
        "pairwise_cos_mean": float(cos.mean().item()),
        "pairwise_cos_std": float(cos.std().item()),
        "pairwise_cos_absmean": float(cos.abs().mean().item()),
        "log_uniformity": uniformity,
    }


def positive_alignment(z_ctx: torch.Tensor, z_tgt: torch.Tensor) -> float:
    """E||z_ctx - z_tgt||^2 over matched samples (JEPA 'positive pairs')."""
    return float((z_ctx - z_tgt).pow(2).sum(-1).mean().item())


# --------------------------------------------------------------------------- #
# Separation metrics
# --------------------------------------------------------------------------- #

def auroc(anom_scores: torch.Tensor, nom_scores: torch.Tensor) -> float:
    """Mann-Whitney AUROC: P(score_anom > score_nom)."""
    a = anom_scores.flatten().double()
    n = nom_scores.flatten().double()
    diff = a[:, None] - n[None, :]
    return float(((diff > 0).double().mean() + 0.5 * (diff == 0).double().mean()).item())


def fit_whitened(Z_ref: torch.Tensor, reg: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = Z_ref.mean(0)
    Zc = Z_ref - mu
    cov = (Zc.T @ Zc) / max(Z_ref.size(0) - 1, 1) + reg * torch.eye(Z_ref.size(1))
    evals, evecs = torch.linalg.eigh(cov)
    prec = evecs @ torch.diag(1.0 / evals.clamp_min(1e-12)) @ evecs.T
    return mu, prec, evecs.flip(-1)  # evecs sorted descending


def maha_dist(Z: torch.Tensor, mu: torch.Tensor, prec: torch.Tensor) -> torch.Tensor:
    d = Z - mu
    return torch.sqrt(torch.clamp((d @ prec * d).sum(-1), min=0.0))


def knn_dist(query: torch.Tensor, ref: torch.Tensor, k: int = 5) -> torch.Tensor:
    d = torch.cdist(query, ref)
    return d.topk(k, largest=False).values.mean(-1)


def separation_block(
    Z_nom: torch.Tensor, Z_anom: torch.Tensor
) -> Dict[str, Any]:
    mu, prec, pcs = fit_whitened(Z_nom)
    d_nom = maha_dist(Z_nom, mu, prec)
    d_anom = maha_dist(Z_anom, mu, prec)
    k_nom = knn_dist(Z_nom, Z_nom, k=6)  # leave-one-out-ish (self-match at 0 still present)
    k_anom = knn_dist(Z_anom, Z_nom, k=5)

    std_nom = Z_nom.std(0, unbiased=False) + 1e-8
    dim_shift = ((Z_anom.mean(0) - Z_nom.mean(0)) / std_nom).abs()
    top_dims = torch.argsort(dim_shift, descending=True)[:8]

    shift_dir = Z_anom.mean(0) - Z_nom.mean(0)
    pc_align = [
        float(torch.nn.functional.cosine_similarity(shift_dir, pcs[i], dim=0).abs().item())
        for i in range(min(5, pcs.size(0)))
    ]

    return {
        "maha_nom_mean": float(d_nom.mean().item()),
        "maha_anom_mean": float(d_anom.mean().item()),
        "maha_auroc": auroc(d_anom, d_nom),
        "knn_nom_mean": float(k_nom.mean().item()),
        "knn_anom_mean": float(k_anom.mean().item()),
        "knn_auroc": auroc(k_anom, k_nom),
        "centroid_shift_maha": float(maha_dist(Z_anom.mean(0, keepdim=True), mu, prec).item()),
        "top_shifted_dims": [
            {"dim": int(i), "shift_sigma": float(dim_shift[i].item())} for i in top_dims
        ],
        "shift_pc_alignment": pc_align,
    }


def residual_block(
    resid_nom: torch.Tensor, resid_anom: torch.Tensor, Z_tgt_nom: torch.Tensor
) -> Dict[str, Any]:
    n_nom = resid_nom.norm(dim=-1)
    n_anom = resid_anom.norm(dim=-1)
    mu, prec, pcs = fit_whitened(Z_tgt_nom)
    w_nom = maha_dist(resid_nom, torch.zeros_like(mu), prec)
    w_anom = maha_dist(resid_anom, torch.zeros_like(mu), prec)

    per_dim_nom = resid_nom.pow(2).mean(0)
    per_dim_anom = resid_anom.pow(2).mean(0)
    dim_gain = (per_dim_anom / (per_dim_nom + 1e-12))
    top_gain = torch.argsort(dim_gain, descending=True)[:8]

    mean_resid_dir = resid_anom.mean(0)
    pc_align = [
        float(torch.nn.functional.cosine_similarity(mean_resid_dir, pcs[i], dim=0).abs().item())
        for i in range(min(5, pcs.size(0)))
    ]

    return {
        "resid_norm_nom": float(n_nom.mean().item()),
        "resid_norm_anom": float(n_anom.mean().item()),
        "resid_norm_ratio": float((n_anom.mean() / (n_nom.mean() + 1e-8)).item()),
        "resid_auroc": auroc(n_anom, n_nom),
        "whitened_resid_auroc": auroc(w_anom, w_nom),
        "top_error_gain_dims": [
            {"dim": int(i), "gain": float(dim_gain[i].item())} for i in top_gain
        ],
        "mean_resid_pc_alignment": pc_align,
    }


# --------------------------------------------------------------------------- #
# Per-model driver
# --------------------------------------------------------------------------- #

def diagnose_model(
    model_key: str,
    display_name: str,
    model: nn.Module,
    ctx_nom: torch.Tensor,
    tgt_nom: torch.Tensor,
    ctx_anom: torch.Tensor,
    tgt_anom: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    model.fit_mahalanobis_covariance(ctx_nom, tgt_nom, batch_size=ctx_nom.size(0))

    nom = extract_latents(model, model_key, ctx_nom, tgt_nom, device)
    anom = extract_latents(model, model_key, ctx_anom, tgt_anom, device)

    disc_kw: Dict[str, Any] = {"use_mahalanobis": True}
    if model_key == "potential_flow":
        disc_kw["include_curvature"] = True
    s_nom = model.compute_predictive_discrepancy(ctx_nom.to(device), tgt_nom.to(device), **disc_kw).cpu()
    s_anom = model.compute_predictive_discrepancy(ctx_anom.to(device), tgt_anom.to(device), **disc_kw).cpu()

    out: Dict[str, Any] = {
        "model_name": display_name,
        "geometry": {
            "z_ctx_nom": {**spectrum_stats(nom["z_ctx"]), **isotropy_stats(nom["z_ctx"])},
            "z_tgt_nom": {**spectrum_stats(nom["z_tgt"]), **isotropy_stats(nom["z_tgt"])},
            "z_pred_nom": {**spectrum_stats(nom["z_pred"]), **isotropy_stats(nom["z_pred"])},
            "z_tgt_anom": {**spectrum_stats(anom["z_tgt"]), **isotropy_stats(anom["z_tgt"])},
        },
        "alignment": {
            "pos_align_nom": positive_alignment(nom["z_ctx"], nom["z_tgt"]),
            "pos_align_anom": positive_alignment(anom["z_ctx"], anom["z_tgt"]),
        },
        "separation_z_tgt": separation_block(nom["z_tgt"], anom["z_tgt"]),
        "residuals": residual_block(nom["residual"], anom["residual"], nom["z_tgt"]),
        "model_score": {
            "score_nom_mean": float(s_nom.mean().item()),
            "score_anom_mean": float(s_anom.mean().item()),
            "score_auroc": auroc(s_anom, s_nom),
        },
    }

    # ---- model-specific extras ----
    extras: Dict[str, Any] = {}
    if model_key == "potential_flow" and model.grassmannian_codebook is not None:
        with torch.no_grad():
            _, probs, _ = model.grassmannian_codebook(nom["z_ctx"].to(device), hard=False)
            probs = probs.cpu()
            hist = probs.argmax(-1).bincount(minlength=probs.size(-1))
            extras["regime_histogram_nom"] = hist.tolist()
            extras["regime_prob_entropy"] = float(
                (-(probs * torch.log(probs + 1e-8)).sum(-1)).mean().item()
            )
    if model_key == "reynolds_stress":
        with torch.no_grad():
            sig_nom = model.compute_observed_stress(nom["z_tgt"].to(device)).cpu()
            sig_anom = model.compute_observed_stress(anom["z_tgt"].to(device)).cpu()
            eig_nom = torch.linalg.eigvalsh(sig_nom).min(-1).values
            eig_anom = torch.linalg.eigvalsh(sig_anom).min(-1).values
            extras["stress_min_eig_nom_mean"] = float(eig_nom.mean().item())
            extras["stress_min_eig_anom_mean"] = float(eig_anom.mean().item())
            extras["stress_cone_violations_nom"] = int((eig_nom < 0).sum().item())
            extras["stress_cone_violations_anom"] = int((eig_anom < 0).sum().item())
    if model_key == "operator_entropy":
        from src.models.jepa.operator_entropy_jepa import von_neumann_entropy
        extras["vn_deficiency_zpred_nom"] = float(von_neumann_entropy(nom["z_pred"].to(device)).item())
        extras["vn_deficiency_zpred_anom"] = float(von_neumann_entropy(anom["z_pred"].to(device)).item())
    out["extras"] = extras

    # persist raw latents for downstream visualization
    npz = {
        f"{k}_nom": v.numpy() for k, v in nom.items()
    }
    npz.update({f"{k}_anom": v.numpy() for k, v in anom.items()})
    npz["score_nom"] = s_nom.numpy()
    npz["score_anom"] = s_anom.numpy()
    return out, npz


def _fmt(d: Dict[str, Any], keys: list) -> str:
    return "  ".join(f"{k}={d[k]:.4g}" if isinstance(d.get(k), float) else f"{k}={d.get(k)}" for k in keys)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Executing Latent Space Diagnosis on device: {device}")

    B = 256
    ctx_nom, tgt_nom, ctx_anom, tgt_anom = generate_synthetic_telemetry(
        batch_size=B, context_len=256, target_len=64, channels=9, seed=42
    )
    channels = ctx_nom.size(-1)

    builders = [
        ("operator_entropy", "OperatorEntropyJEPA", build_operator_entropy),
        ("reynolds_stress", "ReynoldsStressJEPA", build_reynolds_stress),
        ("potential_flow", "PotentialFlowJEPA", build_potential_flow),
    ]

    out_dir = Path("reports/latent_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    for model_key, display_name, builder in builders:
        print("\n" + "=" * 100)
        print(f"LATENT DIAGNOSIS: {display_name}")
        print("=" * 100)

        torch.manual_seed(42)
        model = builder(channels, device)
        res, npz = diagnose_model(
            model_key, display_name, model,
            ctx_nom, tgt_nom, ctx_anom, tgt_anom, device,
        )
        summary[model_key] = res
        np.savez_compressed(out_dir / f"{model_key}_latents.npz", **npz)

        g = res["geometry"]
        for set_name in ("z_ctx_nom", "z_tgt_nom", "z_pred_nom", "z_tgt_anom"):
            s = g[set_name]
            print(f"  [{set_name}] " + _fmt(s, ["mean_norm", "eff_rank", "participation_ratio",
                                                "top1_var_share", "min_eig", "dead_dims_var<1e-6",
                                                "pairwise_cos_absmean", "log_uniformity"]))
        print(f"  alignment: pos_align_nom={res['alignment']['pos_align_nom']:.4f} "
              f"pos_align_anom={res['alignment']['pos_align_anom']:.4f}")

        sep = res["separation_z_tgt"]
        print(f"  separation(z_tgt): maha_auroc={sep['maha_auroc']:.4f} "
              f"knn_auroc={sep['knn_auroc']:.4f} "
              f"centroid_shift={sep['centroid_shift_maha']:.2f} maha")
        print(f"    top shifted dims: " +
              ", ".join(f"d{d['dim']}:{d['shift_sigma']:.1f}sigma" for d in sep["top_shifted_dims"][:5]))
        print(f"    shift->PC alignment: " + ", ".join(f"{a:.3f}" for a in sep["shift_pc_alignment"]))

        r = res["residuals"]
        print(f"  residuals: norm {r['resid_norm_nom']:.4f}->{r['resid_norm_anom']:.4f} "
              f"({r['resid_norm_ratio']:.2f}x)  resid_auroc={r['resid_auroc']:.4f} "
              f"whitened_auroc={r['whitened_resid_auroc']:.4f}")
        print(f"    top error-gain dims: " +
              ", ".join(f"d{d['dim']}:{d['gain']:.1f}x" for d in r["top_error_gain_dims"][:5]))
        print(f"    mean resid->PC alignment: " + ", ".join(f"{a:.3f}" for a in r["mean_resid_pc_alignment"]))

        ms = res["model_score"]
        print(f"  model score: {ms['score_nom_mean']:.4f}->{ms['score_anom_mean']:.4f} "
              f"score_auroc={ms['score_auroc']:.4f}")
        if res["extras"]:
            print(f"  extras: {json.dumps(res['extras'], default=float)}")

    out_file = out_dir / "latent_diagnosis.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLatent diagnosis artifacts saved to: {out_file.as_posix()}")


if __name__ == "__main__":
    main()
