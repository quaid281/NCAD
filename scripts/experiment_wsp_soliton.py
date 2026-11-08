"""Frozen Held-Out Confirmation & Development Evaluation for Weight-Space Programming (Experiments 1, 2, 3).

Evaluates `SolitonReynoldsJEPAModel` (fixed EMA 0.996, batch 16) + the 224-Bit
`MultiPerspectiveCircuitBreakerPanel` across:
  - `dev19`: The 19 development streams (`reports/streams_19_nonghl.txt`).
  - `untouched118`: The 118 held-out streams of `reports/streams_141_tier2.txt` after excluding all
    23 streams in `scripts.circuit_breaker_19stream.common.DEVELOPMENT_STREAMS`.
  - `untouched_non_exathlon`: All 101 non-Exathlon untouched held-out streams across 8 datasets
    (`daphnet`, `msl`, `smap`, `opportunity`, `smd`, `gecco`, `psm`, `creditcard`).

Reports strictly causal (`trailing`) and buffered (`smear`) timeline mappings separately, with
stream-level Wilcoxon tests, 95% stream-bootstrap CIs, 95% dataset-cluster-bootstrap CIs, and
exact dataset-level sign-flip permutation p-values (`scripts.reviewer_analyses.benchmark_141_uncertainty`).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.builder import build_ts_jepa_model
from src.experimental.phase_embedding_synthesis import (
    _extract_subblock_latents,
    _forward_predictors_with_phase,
)
from src.experimental.weight_space_programming import (
    WSPConfig,
    apply_blind_wsp_preconditioning,
    apply_surgical_wsp_quench,
    apply_unified_wsp_soliton_program,
    bake_structural_legendre_phase_basis,
    bind_structural_legendre_phase_basis,
    evaluate_blind_dial_reconstruction,
    synthesize_subblock_phase_embeddings,
)
from src.models._jepa_utils import whiten_residuals
from src.scoring.circuit_breaker_panel import (
    MultiPerspectiveCircuitBreakerPanel,
    stabilize_whitened_subblocks,
)
from scripts.circuit_breaker_19stream.common import DEVELOPMENT_STREAMS
from scripts.reviewer_analyses.benchmark_141_uncertainty import boot_ci, sign_flip_p
from scripts.run_multiseed_heldout_benchmark import (
    batches,
    compute_metrics,
    event_stats,
    load_stream,
)


@torch.no_grad()
def extract_raw4_white4_from_latents(
    model: torch.nn.Module,
    z_ctx: torch.Tensor,
    z_sub_tgt: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (raw4, white4) from cached encoder latents in milliseconds."""
    model.eval()
    M, D = int(model.num_subblocks), int(model.latent_dim)
    z_sub_pred = _forward_predictors_with_phase(model, z_ctx, model.subblock_embeddings)
    raw4 = z_sub_tgt - z_sub_pred
    bsz = raw4.shape[0]
    white4 = whiten_residuals(
        raw4.reshape(bsz, M * D) - model.residual_mean, model.precision_matrix
    ).reshape(bsz, M, D)
    return raw4.cpu().numpy().astype(np.float32), white4.cpu().numpy().astype(np.float32)


def evaluate_model_with_panel_cached(
    model: torch.nn.Module,
    z_ctx_va: torch.Tensor,
    z_tgt_va: torch.Tensor,
    z_ctx_te: torch.Tensor,
    z_tgt_te: torch.Tensor,
    len_va: int,
    len_test: int,
    y_test: np.ndarray,
    context_size: int = 256,
    suspect_size: int = 64,
    apply_eq3_blend: bool = True,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Evaluate a model with the 224-dial circuit-breaker panel under both trailing and smear mappings."""
    panel = MultiPerspectiveCircuitBreakerPanel(context_size=context_size, suspect_size=suspect_size)
    va_raw4, va_white4 = extract_raw4_white4_from_latents(model, z_ctx_va, z_tgt_va)
    te_raw4, te_white4 = extract_raw4_white4_from_latents(model, z_ctx_te, z_tgt_te)
    n_cal_w = len(va_raw4) // 2
    if apply_eq3_blend:
        va_ew, te_ew = stabilize_whitened_subblocks(va_white4, te_white4, va_raw4, te_raw4, n_cal_w)
    else:
        va_ew = np.clip(np.nan_to_num(va_white4.astype(np.float64), nan=0.0), -100.0, 100.0)
        te_ew = np.clip(np.nan_to_num(te_white4.astype(np.float64), nan=0.0), -100.0, 100.0)

    bits = panel.calibrate_and_trip_windows(va_ew, te_ew, n_cal_w)

    out: dict[str, dict[str, float]] = {}
    for mapping in ("trailing", "smear"):
        dec = panel.evaluate_stream_decisions(bits, len_va, len_test, mapping)
        tl = y_test[dec.valid_mask]
        mt = compute_metrics(
            tl,
            dec.test_pred_balanced,
            scores=dec.test_switch_counts,
            valid_mask=np.ones_like(tl, dtype=bool),
            use_pa=False,
        )
        er, _ = event_stats(tl, dec.test_pred_balanced)
        hfpr = float(dec.hold_pred_balanced.mean()) if len(dec.hold_pred_balanced) > 0 else 0.0
        tfpr = float(dec.test_pred_balanced[tl == 0].mean()) if np.any(tl == 0) else 0.0
        out[mapping] = {
            "hold_fpr": hfpr,
            "test_fpr": tfpr,
            "f1": float(mt["f1"]),
            "precision": float(mt["precision"]),
            "recall": float(mt["recall"]),
            "event_recall": float(er),
            "k_gate": float(dec.k_gate_balanced),
        }
    return out, va_white4


def discover_all_mtsbench_streams() -> list[str]:
    """Discover all (dataset:channel) pairs with valid *_train.csv and *_test.csv in mTSBench_data/."""
    bench_root = ROOT / "mTSBench_data"
    found: list[str] = []
    for d in sorted(bench_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        ds = d.name.lower()
        for tr_file in sorted(d.glob("*_train.csv")):
            stem = tr_file.name[:-10]
            if not (d / f"{stem}_test.csv").exists():
                continue
            if stem.lower().startswith(ds + "_"):
                ch = stem[len(ds) + 1 :]
            elif stem.lower() == ds:
                ch = "default"
            else:
                ch = stem
            found.append(f"{ds}:{ch}")
    return found


def select_cohort_streams(cohort: str, streams_file: str, max_streams: int = 0) -> list[str]:
    """Select stream list according to cohort specification and development-set exclusion rules."""
    if cohort == "dev19":
        path = ROOT / "reports" / "streams_19_nonghl.txt"
        streams = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif cohort in ("untouched118", "untouched_non_exathlon", "all141"):
        path = ROOT / "reports" / "streams_141_tier2.txt"
        raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if cohort == "all141":
            streams = raw
        else:
            streams = []
            for s in raw:
                ds, ch = s.split(":")
                if (ds.lower(), ch) in DEVELOPMENT_STREAMS:
                    continue
                if cohort == "untouched_non_exathlon" and ds.lower() == "exathlon":
                    continue
                streams.append(s)
    elif cohort == "all166":
        p141 = ROOT / "reports" / "streams_141_tier2.txt"
        p45 = ROOT / "reports" / "streams_45.txt"
        s141 = [line.strip() for line in p141.read_text(encoding="utf-8").splitlines() if line.strip()]
        s45 = [line.strip() for line in p45.read_text(encoding="utf-8").splitlines() if line.strip()]
        streams = list(dict.fromkeys(s141 + s45))
    elif cohort == "all_mtsbench":
        streams = discover_all_mtsbench_streams()
    else:
        path = Path(streams_file)
        streams = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if max_streams > 0:
        streams = streams[:max_streams]
    return streams


def print_paired_uncertainty_summary(df: pd.DataFrame) -> None:
    """Print stream-level and dataset-cluster uncertainty summary matching benchmark_141_uncertainty.py."""
    rng = np.random.default_rng(0)
    candidate_methods = [
        ("wsp", "Exp1:Cal-WSP"),
        ("blind", "Exp2:ZeroData-WSP"),
        ("scratch", "Exp3a:ScratchLegendre"),
        ("phase128", "Exp3b:Phase128"),
        ("unified", "Exp1+3:Unified-WSP"),
        ("exp4", "Exp4:StructLegendre"),
    ]
    methods = [(p, lbl) for p, lbl in candidate_methods if f"{p}_hold_fpr" in df.columns and df[f"{p}_hold_fpr"].notna().all()]
    sub_tr = df[df["mapping"] == "trailing"]
    print("\n=== BLIND WEIGHT-SPACE & LEGENDRE AUDIT SUMMARY ===")
    print(f"Streams evaluated                           : {len(sub_tr)} across {sub_tr['ds'].nunique()} datasets")
    print(f"Mean 224-Dial Role Reconstruction Accuracy  : {100.0 * sub_tr['exp2_role_acc'].mean():.1f}%")
    print(f"Mean Out-of-Sample Fragility Rank Corr      : {sub_tr['exp2_frag_corr'].mean():+.3f}")
    print(f"Mean SGD Parasitic Saw-Tooth Energy (phi_3) : {sub_tr['exp3_sawtooth_before'].mean():.1f}%")

    for mapping in ("trailing", "smear"):
        sub = df[df["mapping"] == mapping].copy()
        if len(sub) == 0:
            continue
        ds_arr = sub["ds"].to_numpy()
        print(f"\n=== SUMMARY ({mapping.upper()}, n={len(sub)} streams, {len(np.unique(ds_arr))} datasets) ===")
        for metric in ("hold_fpr", "test_fpr", "f1", "event_recall"):
            valid_mask = sub[f"base_{metric}"].notna()
            sub_m = sub[valid_mask]
            ds_m = sub_m["ds"].to_numpy()
            base_v = sub_m[f"base_{metric}"].to_numpy(float)
            lo_b, hi_b = boot_ci(base_v, None, rng)
            print(f"  [Base]                  {metric:12s}: mean={base_v.mean():.4f} (95% CI [{lo_b:.4f}, {hi_b:.4f}])")
            for prefix, label in methods:
                m_v = sub_m[f"{prefix}_{metric}"].to_numpy(float)
                diff = m_v - base_v
                lo_s, hi_s = boot_ci(diff, None, rng)
                lo_d, hi_d = boot_ci(diff, ds_m, rng)
                nonzero = np.abs(diff) > 1e-12
                p_wilc = float(wilcoxon(diff[nonzero]).pvalue) if np.sum(nonzero) >= 2 else 1.0
                ds_means = pd.Series(diff).groupby(ds_m).mean().to_numpy()
                p_sf = sign_flip_p(ds_means) if len(ds_means) >= 2 else 1.0
                print(
                    f"  [{label:21s}] {metric:12s}: mean={m_v.mean():.4f} | d={diff.mean():+.4f} "
                    f"streamCI=[{lo_s:+.4f},{hi_s:+.4f}] dsCI=[{lo_d:+.4f},{hi_d:+.4f}] "
                    f"p_wilc={p_wilc:.4f} p_ds_signflip={p_sf:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen WSP experiments on dev19, untouched held-out, or full archive.")
    parser.add_argument(
        "--cohort",
        type=str,
        default="dev19",
        choices=("dev19", "untouched_non_exathlon", "untouched118", "all141", "all166", "all_mtsbench", "custom"),
        help="Stream cohort to evaluate.",
    )
    parser.add_argument(
        "--streams-file",
        type=str,
        default=str(ROOT / "reports" / "streams_19_nonghl.txt"),
        help="Custom stream list path when --cohort=custom.",
    )
    parser.add_argument("--max-streams", type=int, default=0, help="Optional cap on number of streams.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per stream.")
    parser.add_argument("--include-exp4", action="store_true", help="Also train & evaluate Exp 4 Structural Legendre Basis.")
    parser.add_argument("--no-eq3-blend", action="store_true", help="Disable Eq. (3) stabilize_whitened_subblocks.")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path (supports incremental resume).")
    args = parser.parse_args()

    streams = select_cohort_streams(args.cohort, args.streams_file, args.max_streams)
    C, S, W = 256, 64, 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsp_cfg = WSPConfig(context_len=C, suspect_len=S, candidate_k_counts=(4, 2))
    apply_eq3 = not args.no_eq3_blend

    rows: list[dict] = []
    done_streams: set[str] = set()
    if args.out and Path(args.out).exists():
        existing_df = pd.read_csv(args.out)
        rows = existing_df.to_dict("records")
        done_streams = set(existing_df["stream"].unique())
        print(f"Resuming from {args.out}: {len(done_streams)} streams already completed.")

    for idx_s, s_name in enumerate(streams, start=1):
        if s_name in done_streams:
            continue
        ds, ch = s_name.split(":")
        is_heldout = (ds.lower(), ch) not in DEVELOPMENT_STREAMS
        try:
            train, test, y = load_stream(ds, ch)
        except Exception as exc:
            print(f"[{idx_s:3d}/{len(streams):3d}] {s_name:22s} | SKIPPED load error: {exc}")
            continue
        n_val = int(len(train) * 0.35)
        tr, va = train[:-n_val], train[-n_val:]
        if len(tr) <= W + 16 or len(va) <= W + 200 or len(test) <= W + 64:
            print(f"[{idx_s:3d}/{len(streams):3d}] {s_name:22s} | SKIPPED short stream (tr={len(tr)}, va={len(va)}, te={len(test)})")
            continue
        mu, sd_arr = tr.mean(0), tr.std(0)
        sc = np.where(sd_arr < 1e-8, 1.0, np.maximum(sd_arr, 0.01))
        tr_s, va_s, te_s = (tr - mu) / sc, (va - mu) / sc, (test - mu) / sc
        tr_w = DataLoader.create_windows(tr_s, W, step=16, copy=False)
        va_w = DataLoader.create_windows(va_s, W, step=1, copy=False)
        te_w = DataLoader.create_windows(te_s, W, step=1, copy=False)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cfg = CSMConfig(
            model_type="soliton_reynolds_jepa",
            context_size=C,
            suspect_size=S,
            latent_dim=32,
        )
        model = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            model.train()
            for b in batches(tr_w, 16):
                b = b.to(device)
                loss, _ = model.compute_objective(b[:, :C], b[:, C:], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.update_target_encoder(None)
        model.fit_alien_bank(tr_w, context_len=C)

        z_ctx_tr, z_tgt_tr = _extract_subblock_latents(model, tr_w, C)
        z_ctx_va, z_tgt_va = _extract_subblock_latents(model, va_w, C)
        z_ctx_te, z_tgt_te = _extract_subblock_latents(model, te_w, C)

        base_metrics, va_white4_base = evaluate_model_with_panel_cached(
            model, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )
        n_cal_w = len(va_w) // 2

        # Exp 1: Surgical Zero-Gradient WSP
        model_wsp = copy.deepcopy(model)
        wsp_res = apply_surgical_wsp_quench(model_wsp, tr_w, va_w[:n_cal_w], wsp_cfg)
        wsp_metrics, _ = evaluate_model_with_panel_cached(
            model_wsp, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )

        # Exp 2: 100% Data-Free Reverse Engineering & Pre-Conditioning
        model_blind = copy.deepcopy(model)
        blind_res = apply_blind_wsp_preconditioning(model_blind, k_edit=2, min_eff_rank_abstain=55.0)
        blind_report = blind_res["report"]
        tr_raw4, _ = extract_raw4_white4_from_latents(model, z_ctx_tr, z_tgt_tr)
        blind_eval = evaluate_blind_dial_reconstruction(blind_report, tr_raw4, va_white4_base)
        blind_metrics, _ = evaluate_model_with_panel_cached(
            model_blind, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )

        # Exp 3a: Zero-Shot Scratch Legendre Phase Synthesis (E_0 = 0)
        tr_lat = (z_ctx_tr, z_tgt_tr)
        cal_lat = (z_ctx_va[:n_cal_w], z_tgt_va[:n_cal_w])
        model_scratch = copy.deepcopy(model)
        scratch_res = synthesize_subblock_phase_embeddings(
            model_scratch, tr_w, va_w[:n_cal_w], C, S, from_scratch=True, tr_latents=tr_lat, cal_latents=cal_lat
        )
        scratch_metrics, _ = evaluate_model_with_panel_cached(
            model_scratch, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )

        # Exp 3b: Surgical 128-Parameter Legendre Phase Refinement
        model_phase = copy.deepcopy(model)
        phase_res = synthesize_subblock_phase_embeddings(
            model_phase, tr_w, va_w[:n_cal_w], C, S, from_scratch=False, tr_latents=tr_lat, cal_latents=cal_lat
        )
        phase_metrics, _ = evaluate_model_with_panel_cached(
            model_phase, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )

        # Unified WSP (Exp 1 + Exp 3b Combined Post-Training Program)
        model_uni = copy.deepcopy(model)
        uni_res = apply_unified_wsp_soliton_program(
            model_uni, tr_w, va_w[:n_cal_w], wsp_cfg, tr_latents=tr_lat, cal_latents=cal_lat
        )
        uni_metrics, _ = evaluate_model_with_panel_cached(
            model_uni, z_ctx_va, z_tgt_va, z_ctx_te, z_tgt_te, len(va), len(test), y, C, S, apply_eq3
        )

        # Optional Exp 4: Structural Legendre Horizon Basis During Training + Unified WSP
        exp4_metrics = None
        if args.include_exp4:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model_e4 = build_ts_jepa_model(cfg, input_dim=train.shape[1], device=device)
            bind_structural_legendre_phase_basis(model_e4)
            opt4 = optim.AdamW(model_e4.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(args.epochs):
                model_e4.train()
                for b in batches(tr_w, 16):
                    b = b.to(device)
                    loss, _ = model_e4.compute_objective(b[:, :C], b[:, C:], cfg)
                    opt4.zero_grad(set_to_none=True)
                    loss.backward()
                    opt4.step()
                    model_e4.update_target_encoder(None)
            bake_structural_legendre_phase_basis(model_e4)
            model_e4.fit_alien_bank(tr_w, context_len=C)
            zc_tr4, zt_tr4 = _extract_subblock_latents(model_e4, tr_w, C)
            zc_va4, zt_va4 = _extract_subblock_latents(model_e4, va_w, C)
            zc_te4, zt_te4 = _extract_subblock_latents(model_e4, te_w, C)
            apply_unified_wsp_soliton_program(
                model_e4, tr_w, va_w[:n_cal_w], wsp_cfg,
                tr_latents=(zc_tr4, zt_tr4), cal_latents=(zc_va4[:n_cal_w], zt_va4[:n_cal_w]),
            )
            exp4_metrics, _ = evaluate_model_with_panel_cached(
                model_e4, zc_va4, zt_va4, zc_te4, zt_te4, len(va), len(test), y, C, S, apply_eq3
            )

        for mapping in ("trailing", "smear"):
            bm = base_metrics[mapping]
            wm = wsp_metrics[mapping]
            blm = blind_metrics[mapping]
            scm = scratch_metrics[mapping]
            phm = phase_metrics[mapping]
            unm = uni_metrics[mapping]
            row_dict = {
                "stream": s_name,
                "ds": ds.lower(),
                "ch": ch,
                "heldout": is_heldout,
                "mapping": mapping,
                "exp1_mode": wsp_res.mode,
                "exp2_regime": blind_report.dynamical_regime,
                "exp2_role_acc": blind_eval["role_accuracy"],
                "exp2_frag_corr": blind_eval["fragility_rank_corr"],
                "exp2_mode": blind_res["mode"],
                "exp3_sawtooth_before": phase_res.sawtooth_energy_pct_before,
                "exp3_mode": phase_res.mode,
                "unified_mode": uni_res["mode"],
                "base_hold_fpr": bm["hold_fpr"],
                "wsp_hold_fpr": wm["hold_fpr"],
                "blind_hold_fpr": blm["hold_fpr"],
                "scratch_hold_fpr": scm["hold_fpr"],
                "phase128_hold_fpr": phm["hold_fpr"],
                "unified_hold_fpr": unm["hold_fpr"],
                "base_test_fpr": bm["test_fpr"],
                "wsp_test_fpr": wm["test_fpr"],
                "blind_test_fpr": blm["test_fpr"],
                "scratch_test_fpr": scm["test_fpr"],
                "phase128_test_fpr": phm["test_fpr"],
                "unified_test_fpr": unm["test_fpr"],
                "base_f1": bm["f1"],
                "wsp_f1": wm["f1"],
                "blind_f1": blm["f1"],
                "scratch_f1": scm["f1"],
                "phase128_f1": phm["f1"],
                "unified_f1": unm["f1"],
                "base_event_recall": bm["event_recall"],
                "wsp_event_recall": wm["event_recall"],
                "blind_event_recall": blm["event_recall"],
                "scratch_event_recall": scm["event_recall"],
                "phase128_event_recall": phm["event_recall"],
                "unified_event_recall": unm["event_recall"],
            }
            if exp4_metrics is not None:
                e4m = exp4_metrics[mapping]
                row_dict["exp4_hold_fpr"] = e4m["hold_fpr"]
                row_dict["exp4_test_fpr"] = e4m["test_fpr"]
                row_dict["exp4_f1"] = e4m["f1"]
                row_dict["exp4_event_recall"] = e4m["event_recall"]
            rows.append(row_dict)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.csv")
            pd.DataFrame(rows).to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)

        b_tr = base_metrics["trailing"]
        w_tr = wsp_metrics["trailing"]
        bl_tr = blind_metrics["trailing"]
        ph_tr = phase_metrics["trailing"]
        un_tr = uni_metrics["trailing"]
        e4_str = f" E4={exp4_metrics['trailing']['test_fpr']:.4f}" if exp4_metrics is not None else ""
        print(
            f"[{idx_s:3d}/{len(streams):3d}] {s_name:22s} | Role={100*blind_eval['role_accuracy']:4.1f}% "
            f"Saw={phase_res.sawtooth_energy_pct_before:4.1f}% | "
            f"tFPR: Base={b_tr['test_fpr']:.4f} WSP={w_tr['test_fpr']:.4f} Blind={bl_tr['test_fpr']:.4f} "
            f"Ph128={ph_tr['test_fpr']:.4f} Uni={un_tr['test_fpr']:.4f}{e4_str} | "
            f"F1: Base={b_tr['f1']:.4f} Uni={un_tr['f1']:.4f}"
        )

    df = pd.DataFrame(rows)
    print_paired_uncertainty_summary(df)


if __name__ == "__main__":
    main()
