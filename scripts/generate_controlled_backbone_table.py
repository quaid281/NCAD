"""Generate LaTeX table for the controlled backbone experiment.
Matches exact HybridTCNEncoder architecture across observation reconstruction, observation forecasting, and latent prediction.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "reports" / "controlled_backbone_experiment.csv"
OUT_TEX = ROOT / "paper" / "NCAD_CS" / "sections" / "tab_controlled_backbone.tex"

def main():
    df = pd.read_csv(CSV_PATH)
    
    clean = df[df["snr_db"].astype(str) == "inf"]
    noisy_20 = df[df["snr_db"] == 20.0]
    noisy_10 = df[df["snr_db"] == 10.0]
    noisy_0 = df[df["snr_db"] == 0.0]
    
    models = [
        ("tcn_obs_recon", "TCN Observation Autoencoder (Reconstruction)"),
        ("tcn_obs_pred", "TCN Observation Forecaster (Observation-Space JEPA)"),
        ("ts_jepa", "TS-JEPA (Latent-Space Predictive JEPA)"),
    ]
    
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Matched-Backbone Controlled Ablation across 15 Telemetry Streams. All three models use the exact same \texttt{HybridTCNEncoder} backbone ($F=48, D=32, C=256, S=64$), identical training epochs (20), AdamW optimizer, and identical held-out EVT/SPOT validation calibration. Observation forecasting in raw sensor space matches reconstruction behavior ($\kappa \approx 56$, $\text{FPR} \approx 27.6\%$, $p=0.68$). Shifting the predictive loss to latent space compresses residual tails ($\kappa = 1.77$), significantly lowering clean false-alarm rate ($p=0.048$) and stabilizing against severe sensor noise ($0\,\text{dB}$ FPR $13.9\%$ vs.\ $51.0\%$, $p=0.0026$), at the cost of lower point-level recall.}")
    lines.append(r"\label{tab:controlled_backbone}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc} \toprule")
    lines.append(r"\textbf{Architecture \& Objective} & \textbf{Val. Kurtosis} & \textbf{Test Kurtosis} & \textbf{Clean FPR} & \textbf{0\,dB FPR} & \textbf{Clean Point-F1} & \textbf{Clean Recall} \\")
    lines.append(r" & ($\kappa_{\text{val}}$) & ($\kappa_{\text{test}}$) & ($\text{SNR}=\infty$) & ($\text{SNR}=0\,\text{dB}$) & ($\text{SNR}=\infty$) & ($\text{SNR}=\infty$) \\ \midrule")
    
    for m_key, m_label in models:
        c_sub = clean[clean["model"] == m_key]
        n_sub = noisy_0[noisy_0["model"] == m_key]
        
        vk_m, vk_s = c_sub["val_kurtosis"].mean(), c_sub["val_kurtosis"].std()
        tk_m, tk_s = c_sub["test_kurtosis"].mean(), c_sub["test_kurtosis"].std()
        cfpr_m, cfpr_s = c_sub["empirical_fpr"].mean(), c_sub["empirical_fpr"].std()
        nfpr_m, nfpr_s = n_sub["empirical_fpr"].mean(), n_sub["empirical_fpr"].std()
        cf1_m, cf1_s = c_sub["point_f1"].mean(), c_sub["point_f1"].std()
        crec_m, crec_s = c_sub["point_rec"].mean(), c_sub["point_rec"].std()
        
        lines.append(
            f"{m_label} & "
            f"${vk_m:.2f} \\pm {vk_s:.2f}$ & "
            f"${tk_m:.2f} \\pm {tk_s:.2f}$ & "
            f"${cfpr_m:.4f} \\pm {cfpr_s:.3f}$ & "
            f"${nfpr_m:.4f} \\pm {nfpr_s:.3f}$ & "
            f"${cf1_m:.4f} \\pm {cf1_s:.3f}$ & "
            f"${crec_m:.4f} \\pm {crec_s:.3f}$ \\\\"
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
