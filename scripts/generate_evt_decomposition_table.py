"""Generate LaTeX table for the EVT Calibration Failure Decomposition.
Isolates autocorrelation, distribution drift, GPD goodness of fit, and risk overshoot.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "reports" / "evt_failure_decomposition.csv"
OUT_TEX = ROOT / "paper" / "NCAD_CS" / "sections" / "tab_evt_failure_decomposition.tex"

def main():
    df = pd.read_csv(CSV_PATH)
    
    summary = df.groupby("model").agg({
        "rho1_autocorr": ["mean", "std"],
        "extremal_index_theta": ["mean", "std"],
        "mean_cluster_size": ["mean", "std"],
        "ks_stat_shift": ["mean", "std"],
        "delta_q98_pct": ["mean", "std"],
        "held_out_nominal_fpr": ["mean", "std"],
        "overshoot_ratio": ["mean", "std"],
    })
    
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Empirical Decomposition of Extreme Value Calibration Failure. We isolate the four hypothesized drivers of threshold breakdown across benchmark telemetry streams: (1) residual autocorrelation ($\rho_1$) and extremal clustering ($\theta, 1/\theta$), which reduce the effective independent tail sample size ($N_{\text{eff}} = \theta N_u$); (2) non-stationary distribution drift between calibration and held-out evaluation splits (Kolmogorov-Smirnov $D_{\text{KS}}$ and relative 98th-percentile drift $\Delta q_{98}\%$); and (3) actual held-out nominal False-Positive Rate and empirical risk overshoot relative to the theoretical design risk $q=10^{-3}$ ($0.1\%$). Observation-space models exhibit large distribution drift ($\Delta q_{98} = +470\%$) and severe clustering ($1/\theta \approx 12.7\text{--}13.3$ steps), causing a $43\times$ to $47\times$ overshoot of design risk. In contrast, latent prediction compresses distribution drift ($\Delta q_{98} = -0.36\%$), reducing risk overshoot by an order of magnitude.}")
    lines.append(r"\label{tab:evt_failure_decomposition}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc} \toprule")
    lines.append(r"\textbf{Architecture \& Objective} & \textbf{Autocorr. ($\rho_1$)} & \textbf{Extremal Index ($\theta$)} & \textbf{Mean Cluster} & \textbf{KS Drift ($D_{\text{KS}}$)} & \textbf{Held-Out FPR} & \textbf{Risk Overshoot} \\")
    lines.append(r" & (Lag-1) & (Ferro-Segers) & ($1/\theta$ steps) & (Cal vs.\ Eval) & (Nominal Target: $0.1\%$) & ($\text{FPR}_{\text{eval}} / 10^{-3}$) \\ \midrule")
    
    models = [
        ("TimesNet", "TimesNet (Reconstructive)"),
        ("TranAD", "TranAD (Reconstructive)"),
        ("TS-JEPA", "TS-JEPA (Latent Predictive)"),
    ]
    
    for m_key, m_label in models:
        m_df = df[df["model"] == m_key]
        r1_m, r1_s = m_df["rho1_autocorr"].mean(), m_df["rho1_autocorr"].std()
        th_m, th_s = m_df["extremal_index_theta"].mean(), m_df["extremal_index_theta"].std()
        cl_m, cl_s = m_df["mean_cluster_size"].mean(), m_df["mean_cluster_size"].std()
        ks_m, ks_s = m_df["ks_stat_shift"].mean(), m_df["ks_stat_shift"].std()
        fpr_m, fpr_s = m_df["held_out_nominal_fpr"].mean(), m_df["held_out_nominal_fpr"].std()
        ov_m, ov_s = m_df["overshoot_ratio"].mean(), m_df["overshoot_ratio"].std()
        
        lines.append(
            f"{m_label} & "
            f"${r1_m:.3f} \\pm {r1_s:.3f}$ & "
            f"${th_m:.3f} \\pm {th_s:.3f}$ & "
            f"${cl_m:.1f} \\pm {cl_s:.1f}$ & "
            f"${ks_m:.3f} \\pm {ks_s:.3f}$ & "
            f"${fpr_m*100:.2f}\\% \\pm {fpr_s*100:.2f}\\%$ & "
            f"${ov_m:.1f}\\times \\pm {ov_s:.1f}\\times$ \\\\"
        )
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    tex_content = "\n".join(lines) + "\n"
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"Wrote table to {OUT_TEX}")

if __name__ == "__main__":
    main()
