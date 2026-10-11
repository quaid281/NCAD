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
             r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"Model & $\gamma_2$ & $q_{99.9}/q_{99}$ & $\mathrm{ACF}_1$ & $\theta$ & "
             r"$N_{\mathrm{eff}}$ & FPR & PR-AUC \\", r"\midrule"]
    for m, r in g.iterrows():
        lines.append(f"{m} & {fmt(r['kurt'],2)} & {fmt(r.qr)} & {fmt(r.acf1)} & "
                     f"{fmt(r.theta)} & {fmt(r.neff,1)} & {fmt(r.hold)} & {fmt(r.prauc)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
    labels = {"spot": "SPOT", "spot_decluster": "SPOT+decl.", "empirical": "p99.5",
              "rolling_emp": "Roll.\ p99.5", "rolling_spot": "Roll.\ SPOT"}
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


def main():
    rd = pd.read_csv(ROOT / "reports" / "residual_diagnostics.csv")
    mf = pd.read_csv(ROOT / "reports" / "matched_fpr_evaluation.csv")
    dc = pd.read_csv(ROOT / "reports" / "controlled_drift_calibration.csv")
    npdf = pd.read_csv(ROOT / "reports" / "heldout_nonparametric.csv")
    tex = ["\\section{Revision Experiments}\\label{sup:revision_experiments}",
           "",
           "This section collects the four analyses added in response to the review: "
           "matched-backbone residual diagnostics, detection quality at matched "
           "false-positive-rate budgets, a controlled-drift calibration study, and "
           "training-free nonparametric references.",
           "",
           residual_table(rd), "", matched_fpr_table(mf), "",
           drift_table(dc), "", nonparam_table(npdf)]
    OUT.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
