"""Generate supplementary LaTeX tables for the Round-3 revision experiments.

Consumes:
  reports/residual_diagnostics.csv       -> Table: matched-backbone residual diagnostics
  reports/matched_fpr_evaluation.csv     -> Table: detection metrics at matched FPR budgets
  reports/controlled_drift_calibration.csv -> Table: calibrator behavior under controlled shifts
  reports/heldout_nonparametric.csv      -> Table: nonparametric baseline results

Writes: paper/NCAD_CS/sections/sup_revision_experiments.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "NCAD_CS" / "sections" / "sup_revision_experiments.tex"


def fmt(x, d=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return f"{v:.{d}f}" if np.isfinite(v) else "--"


def residual_table(df):
    """Stream-averaged residual diagnostics per model (buffered mapping)."""
    d = df[df.mapping == "smear"].copy()
    for c in ("kurtosis", "tail_ratio", "acf1", "extremal_theta", "n_eff",
              "hold_fpr", "test_pr_auc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby("model").agg(
        kurt=("kurtosis", "mean"), qr=("tail_ratio", "mean"),
        acf1=("acf1", "mean"), theta=("extremal_theta", "mean"),
        neff=("n_eff", "mean"), hold=("hold_fpr", "mean"),
        prauc=("test_pr_auc", "mean"))
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Residual-distribution diagnostics on held-out nominal telemetry "
             r"(buffered regime, stream-averaged). $\gamma_2$: excess kurtosis; "
             r"$q_{99.9}/q_{99}$: tail ratio; $\theta$: extremal index "
             r"(runs declustering); $N_{\mathrm{eff}}$: effective independent exceedances; "
             r"FPR: out-of-sample exceedance rate at the SPOT threshold; PR-AUC: anomaly-vs-nominal "
             r"separation on labeled test data.}\label{tab:residual_diagnostics}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"Model & $\gamma_2$ & $q_{99.9}/q_{99}$ & $\mathrm{ACF}_1$ & $\theta$ & "
             r"$N_{\mathrm{eff}}$ & FPR & PR-AUC \\", r"\midrule"]
    for m, r in g.iterrows():
        lines.append(f"{m} & {fmt(r['kurt'],2)} & {fmt(r.qr)} & {fmt(r.acf1)} & "
                     f"{fmt(r.theta)} & {fmt(r.neff,1)} & {fmt(r.hold)} & {fmt(r.prauc)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def matched_fpr_table(df):
    """Metrics at matched empirical-FPR budgets (buffered + causal)."""
    d = df[df.calibrator.str.startswith("q*=")].copy()
    d["budget"] = d.calibrator.str.replace("q*=", "", regex=False).astype(float)
    rows = []
    for (mapping, budget), g in d.groupby(["mapping", "budget"]):
        for m, r in g.groupby("model").agg(
                er=("event_recall", "mean"), prec=("point_precision", "mean"),
                delay=("mean_delay", "mean"), far=("far_per_1k", "mean"),
                credit=("credit_event_recall", "mean")).iterrows():
            rows.append((mapping, budget, m, r.er, r.prec, r.delay, r.far, r.credit))
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Detection quality at matched held-out false-positive-rate budgets "
             r"(stream-averaged, $q^*$ = empirical calibration quantile). ER: event recall; "
             r"P: point precision; $\Delta t$: mean detection delay; FA/1k: false alarms per "
             r"1{,}000 nominal points; ER$_{\mathrm{pre}}$: recall crediting pre-onset alarms.}"
             r"\label{tab:matched_fpr}",
             r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Regime ($q^*$) & Model & ER & P & $\Delta t$ & FA/1k & ER$_{\mathrm{pre}}$ \\",
             r"\midrule"]
    for mapping, budget, m, er, prec, delay, far, credit in rows:
        reg = f"{'Buffered' if mapping=='smear' else 'Causal'} {budget:.3f}"
        lines.append(f"{reg} & {m} & {fmt(er)} & {fmt(prec)} & {fmt(delay,1)} & "
                     f"{fmt(far,1)} & {fmt(credit)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def drift_table(df):
    """Calibrator x condition grid of achieved FPR."""
    g = df.groupby(["condition", "calibrator"]).achieved_fpr.mean().unstack()
    cols = ["spot", "spot_decluster", "empirical", "rolling_emp", "rolling_spot"]
    cols = [c for c in cols if c in g.columns]
    labels = {"spot": "SPOT", "spot_decluster": "SPOT+decl.", "empirical": r"Empirical ($q=10^{-3}$)",
              "rolling_emp": r"Roll.\ Emp.\ ($q=10^{-3}$)", "rolling_spot": r"Roll.\ SPOT"}
    n_str = df.groupby(["dataset", "channel"]).ngroups
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Controlled-drift calibration study: achieved held-out FPR (design "
             r"$q=10^{-3}$) under injected shifts in real residual streams (stream-averaged "
             rf"over {n_str} streams with $\ge 200$ calibration/evaluation windows; the "
             r"three short MSL streams are excluded). Rolling variants recalibrate per "
             r"evaluation block.}\label{tab:drift_calibration}",
             r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}", r"\toprule",
             "Condition & " + " & ".join(labels[c] for c in cols) + r" \\", r"\midrule"]
    for cond, r in g.iterrows():
        lines.append(f"{cond.replace('_', ' ')} & " +
                     " & ".join(fmt(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def nonparam_table(df):
    """Nonparametric baselines vs headline models (buffered)."""
    d = df[(df.mapping == "smear") & (df.calibrator == "SPOT")]
    g = d.groupby("model").agg(fpr=("hold_fpr", "mean"), f1=("f1", "mean"),
                               prauc=("pr_auc", "mean"), er=("event_recall", "mean"),
                               delay=("mean_delay", "mean"))
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Training-free nonparametric references under the identical held-out "
             r"calibration protocol (19 non-GHL streams, five seeds, buffered regime).}"
             r"\label{tab:nonparametric}",
             r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & Held-out FPR & Point-F1 & PR-AUC & Event Recall & $\Delta t$ \\",
             r"\midrule"]
    for m, r in g.iterrows():
        lines.append(f"{m} & {fmt(r.fpr)} & {fmt(r.f1)} & {fmt(r.prauc)} & "
                     f"{fmt(r.er)} & {fmt(r.delay,1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def oc_table():
    """Operating-characteristic summary: partial AUC + q*=0.05 realized metrics."""
    p = pd.read_csv(ROOT / "reports" / "oc_partial_auc.csv")
    g = (p.groupby(["model", "mapping"])
          .agg(pauc=("partial_auc", "mean"), er=("event_recall_q05", "mean"))
          .reset_index())
    c = pd.read_csv(ROOT / "reports" / "oc_model_curves.csv")
    rf = (c[np.isclose(c.q, 0.05) & (c.cohort == "all")]
          .set_index(["model", "mapping"]).realized_fpr)
    g = g.set_index(["model", "mapping"]).join(rf).reset_index()
    order = g[(g.mapping == "trailing")].sort_values("pauc", ascending=False).model
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Operating-characteristic summary on 19 non-GHL streams "
             r"(seeds averaged within streams). Partial AUC = mean event recall over "
             r"design budgets $\le 0.10$; $q^*{=}0.05$ columns give event recall and "
             r"\emph{realized} test FPR at that design budget.}"
             r"\label{tab:operating_characteristic}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Model & pAUC (trail.) & pAUC (smear) & ER$_{0.05}$ (trail.) & "
             r"FPR$_{0.05}$ (trail.) & ER$_{0.05}$ (smear) & FPR$_{0.05}$ (smear) \\",
             r"\midrule"]
    for m in order:
        t = g[(g.model == m) & (g.mapping == "trailing")].iloc[0]
        s = g[(g.model == m) & (g.mapping == "smear")].iloc[0]
        lines.append(f"{m} & {fmt(t.pauc)} & {fmt(s.pauc)} & {fmt(t.er)} & "
                     f"{fmt(t.realized_fpr)} & {fmt(s.er)} & {fmt(s.realized_fpr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def noise_table():
    pf = pd.read_csv(ROOT / "reports" / "pf_noise_path_test.csv")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Potential-Flow inference-path test: scoring at the "
             r"$z_{\text{noise}}{=}0$ mean path vs.\ noise-matched querying "
             r"($z{=}0.5(z_{\text{noise}}{+}z_{\text{tgt}})$, 16 draws). Noise "
             r"matching does not rescue performance, rejecting the "
             r"train/test-mismatch explanation.}\label{tab:pf_noise}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Stream & Mapping & Path & FPR & Event Recall \\", r"\midrule"]
    for _, r in pf.iterrows():
        path = "noise-matched" if r.noise_matched else "mean ($z_n{=}0$)"
        lines.append(f"{r.dataset}:{r.channel} & {r.mapping} & {path} & "
                     f"{fmt(r.realized_fpr)} & {fmt(r.event_recall)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def trivial_delay_table():
    t = pd.read_csv(ROOT / "reports" / "trivial_f1_comparison.csv")
    g = t.groupby("model").agg(beat=("beats_trivial", "mean"),
                               mf1=("model_f1", "mean"),
                               triv=("trivial_f1", "mean"))
    d = pd.read_csv(ROOT / "reports" / "delay_intersection.csv")
    dg = d.groupby("model")[["delay_main", "delay_other"]].mean()
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{(a) Fraction of stream-seed cells where each model's "
             r"Point-F1 exceeds the trivial all-positive F1 $2\pi/(1{+}\pi)$ at "
             r"$q^*{=}0.05$; (b) mean detection delay on the intersection of "
             r"events detected by both the model and TS-JEPA (steps).}"
             r"\label{tab:trivial_delay}",
             r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Model & $P(\mathrm{F1} > \mathrm{trivial})$ & Point-F1 & "
             r"$\Delta t$ model & $\Delta t$ TS-JEPA \\", r"\midrule"]
    for m, r in g.iterrows():
        dl = dg.loc[m] if m in dg.index else None
        dm = f"{dl.delay_other:.1f}" if dl is not None else "--"
        dj = f"{dl.delay_main:.1f}" if dl is not None else "--"
        lines.append(f"{m} & {fmt(r.beat,2)} & {fmt(r.mf1)} & {dm} & {dj} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def drift_link_table():
    l = pd.read_csv(ROOT / "reports" / "tail_calibration_link.csv")
    from scipy import stats
    rho_k = stats.spearmanr(l.cal_kurtosis, l.realized_fpr, nan_policy="omit")
    rho_d = stats.spearmanr(l.ks_drift, l.realized_fpr, nan_policy="omit")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Stream-level drivers of realized SPOT FPR across all "
             r"dumped cells (Spearman $\rho$): calibration$\to$test KS drift "
             r"predicts failure; calibration-block kurtosis is mildly protective.}"
             r"\label{tab:drift_link}",
             r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Predictor & $\rho$ vs.\ realized FPR & $p$ \\", r"\midrule",
             f"KS drift (cal $\\to$ test) & {rho_k.statistic if False else rho_d.statistic:.3f} & {rho_d.pvalue:.1e} \\\\",
             f"Cal-block excess kurtosis & {rho_k.statistic:.3f} & {rho_k.pvalue:.1e} \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def sota_baselines_table():
    df = pd.read_csv(ROOT / "reports" / "all_models_multidataset_benchmark.csv")
    common = ['CalIt2', 'cicids', 'Daphnet', 'GECCO', 'Genesis', 'GHL', 'metro', 'MSL', 'room-occupancy', 'SMAP', 'swan']
    models = ['ncad', 'dcdetector', 'anomaly_transformer', 'timesnet', 'tranad', 'ts_jepa']
    df_c = df[df.dataset.isin(common) & df.model.isin(models)]
    p_f1 = df_c.groupby(['dataset', 'model'])['point_f1'].mean().unstack()[models]
    pa_f1 = df_c.groupby(['dataset', 'model'])['pa_f1'].mean().unstack()[models]
    counts = df_c.groupby('dataset')['channel'].nunique()

    dataset_labels = {
        'CalIt2': 'CalIt2', 'cicids': 'CICIDS', 'Daphnet': 'Daphnet', 'GECCO': 'GECCO',
        'Genesis': 'Genesis', 'GHL': 'GHL', 'metro': 'Metro', 'MSL': 'MSL',
        'room-occupancy': 'Room-Occ.', 'SMAP': 'SMAP', 'swan': 'SWAN'
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Multi-dataset benchmark comparison for published deep anomaly detection baselines across 11 telemetry datasets (130 streams; unified pipeline under SPOT calibration). Panel A reports unadjusted Point-F1; Panel B reports Point-Adjusted F1 (PA-F1). Macro averages weight each dataset equally; Stream-Macro averages across all 130 evaluation streams. Best result in \textbf{bold}, second \underline{underlined}.}\label{tab:sota_baselines}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"Dataset & Streams & NCAD & DCdetector & Anom.Trans. & TimesNet & TranAD & TS-JEPA \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel A: Unadjusted Point-F1}} \\"
    ]

    for d in common:
        vals = [p_f1.loc[d, m] for m in models]
        max_val = max(vals)
        second_val = sorted(vals, reverse=True)[1] if len(vals) > 1 else -1
        cells = []
        for v in vals:
            s = f"{v:.3f}"
            if v == max_val and v > 0:
                s = r"\textbf{" + s + "}"
            elif v == second_val and v > 0:
                s = r"\underline{" + s + "}"
            cells.append(s)
        lines.append(f"{dataset_labels[d]:12s} & {counts[d]:2d} & " + " & ".join(cells) + r" \\")

    macro_vals = [p_f1[m].mean() for m in models]
    max_m = max(macro_vals)
    second_m = sorted(macro_vals, reverse=True)[1]
    m_cells = [(r"\textbf{" + f"{v:.3f}" + "}" if v == max_m else r"\underline{" + f"{v:.3f}" + "}" if v == second_m else f"{v:.3f}") for v in macro_vals]
    lines.append(r"\midrule")
    lines.append(r"\textbf{Macro Mean}  & 11  & " + " & ".join(m_cells) + r" \\")

    stream_vals = [df_c[df_c.model==m]['point_f1'].mean() for m in models]
    max_s = max(stream_vals)
    second_s = sorted(stream_vals, reverse=True)[1]
    s_cells = [(r"\textbf{" + f"{v:.3f}" + "}" if v == max_s else r"\underline{" + f"{v:.3f}" + "}" if v == second_s else f"{v:.3f}") for v in stream_vals]
    lines.append(r"\textbf{Stream Mean} & 130 & " + " & ".join(s_cells) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Panel B: Point-Adjusted F1 (PA-F1)}} \\")

    for d in common:
        vals = [pa_f1.loc[d, m] for m in models]
        max_val = max(vals)
        second_val = sorted(vals, reverse=True)[1] if len(vals) > 1 else -1
        cells = []
        for v in vals:
            s = f"{v:.3f}"
            if v == max_val and v > 0:
                s = r"\textbf{" + s + "}"
            elif v == second_val and v > 0:
                s = r"\underline{" + s + "}"
            cells.append(s)
        lines.append(f"{dataset_labels[d]:12s} & {counts[d]:2d} & " + " & ".join(cells) + r" \\")

    macro_vals = [pa_f1[m].mean() for m in models]
    max_m = max(macro_vals)
    second_m = sorted(macro_vals, reverse=True)[1]
    m_cells = [(r"\textbf{" + f"{v:.3f}" + "}" if v == max_m else r"\underline{" + f"{v:.3f}" + "}" if v == second_m else f"{v:.3f}") for v in macro_vals]
    lines.append(r"\midrule")
    lines.append(r"\textbf{Macro Mean}  & 11  & " + " & ".join(m_cells) + r" \\")

    stream_vals = [df_c[df_c.model==m]['pa_f1'].mean() for m in models]
    max_s = max(stream_vals)
    second_s = sorted(stream_vals, reverse=True)[1]
    s_cells = [(r"\textbf{" + f"{v:.3f}" + "}" if v == max_s else r"\underline{" + f"{v:.3f}" + "}" if v == second_s else f"{v:.3f}") for v in stream_vals]
    lines.append(r"\textbf{Stream Mean} & 130 & " + " & ".join(s_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table}",
        "",
        r"\subsection{Multi-Dataset Baseline Benchmark Analysis}",
        r"Table~\ref{tab:sota_baselines} evaluates TS-JEPA against published deep anomaly detection baselines across all 11 shared benchmark datasets (130 continuous telemetry streams) using the unified out-of-sample calibration protocol (standard SPOT, $u=0.98, q=10^{-3}$). Two empirical findings stand out:",
        "",
        r"\emph{Contrastive sensitivity under tail calibration}: NCAD \cite{carmona2021neural}, which trains against synthetic point and contextual anomalies, achieves respectable Point-Adjusted F1 ($0.252$ macro, $0.298$ stream-level) but collapses to $0.058$ macro Point-F1 ($0.088$ stream-level). Point adjustment masks rampant false alarms because an isolated trigger within a long anomaly awards full segment credit. When evaluated strictly point-by-point, synthetic anomalies fail to transfer to physical process deviations (yielding near-zero detection on GHL, Genesis, and Metro). Similarly, dual-attention discrepancy contrasting in DCdetector \cite{yang2023dcdetector} and Anomaly Transformer \cite{xu2022anomaly} collapses to $0.070$ and $0.091$ macro Point-F1 due to severe tail-threshold overshoots on non-stationary baselines.",
        "",
        r"\emph{Forecasting versus latent representation}: Observation-space forecasters (TimesNet \cite{wu2023timesnet}: $0.129$ macro Point-F1; TranAD \cite{tuli2022tranad}: $0.156$) outperform contrastive baselines on point metrics but remain constrained by raw observation noise. TS-JEPA achieves leading macro Point-F1 ($0.160$) and stream-level Point-F1 ($0.186$), alongside leading Point-Adjusted F1 ($0.310$ macro, $0.401$ stream-level), indicating that predicting representations in latent space preserves fault dynamics across disparate physical domains without synthetic anomaly synthesis."
    ]
    return "\n".join(lines)


def main():
    rd = pd.read_csv(ROOT / "reports" / "residual_diagnostics.csv")
    mf = pd.read_csv(ROOT / "reports" / "matched_fpr_evaluation.csv")
    dc = pd.read_csv(ROOT / "reports" / "controlled_drift_calibration.csv")
    npdf = pd.concat(
        [pd.read_csv(ROOT / "reports" / "heldout_nonparametric.csv"),
         pd.read_csv(ROOT / "reports" / "heldout_19stream_r4_arms.csv")],
        ignore_index=True)
    tex = ["\\section{Revision Experiments}\\label{sup:revision_experiments}",
           "",
           "This section collects the analyses added in response to review: "
           "matched-backbone and score-space-matched residual diagnostics, "
           "detection quality on the operating-characteristic grid, a "
           "controlled-drift calibration study, training-free and distance-based "
           "references, detection-delay intersection, trivial-baseline "
           "comparison, drift attribution, the Potential-Flow "
           "inference-path test, and multi-dataset comparisons against "
           "published deep anomaly detection baselines.",
           "",
           residual_table(rd), "", oc_table(), "", matched_fpr_table(mf), "",
           drift_table(dc), "", nonparam_table(npdf), "",
           trivial_delay_table(), "", drift_link_table(), "", noise_table(), "",
           sota_baselines_table()]
    OUT.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

