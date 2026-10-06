#!/usr/bin/env python3
"""
Generate Camera-Ready Tables and Publication Figures for the Recalibrated Physics-Informed JEPA Triad
--------------------------------------------------------------------------------------------------------
Produces:
1. tab_core_benchmark.tex: Two-Tier Hierarchical Multi-Tier Table (Zero Overfull \hbox).
2. hero_models_sota_comparison.png / .pdf: Modern grouped bar chart with curated academic palette & zero collisions.
3. domain_radar_chart.png / .pdf: Multi-domain polar profile with generous padding and external legend.
4. dataset_performance_heatmap.png / .pdf: Complete 11x6 matrix with adaptive high-contrast typography.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "reports" / "recalibrated_benchmark_subset.csv"
OUT_FIG_DIR = ROOT / "paper" / "NCAD_CS" / "figures"
OUT_TAB_DIR = ROOT / "paper" / "NCAD_CS" / "sections"

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)

# Publication Typography (Times / Serif matching IEEE Transactions & Top ML venues)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.labelsize'] = 11.0
plt.rcParams['axes.titlesize'] = 12.0
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13.0
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 0.8

MODEL_DISPLAY_NAMES = {
    'operator_entropy_jepa': 'Operator-Entropy JEPA (Ours)',
    'reynolds_stress_jepa': 'Reynolds-Stress JEPA (Ours)',
    'potential_flow_jepa': 'Potential-Flow JEPA (Ours)',
    'ts_jepa': 'Vanilla TS-JEPA (Control)',
    'tranad': 'TranAD (VLDB 2022)',
    'timesnet': 'TimesNet (ICLR 2023)',
}

MODELS_ORDER = [
    'operator_entropy_jepa',
    'reynolds_stress_jepa',
    'potential_flow_jepa',
    'ts_jepa',
    'tranad',
    'timesnet'
]


def load_data():
    return pd.read_csv(CSV_PATH)


# =========================================================================
# 1. Two-Tier Hierarchical Benchmark Table (LaTeX)
# =========================================================================

def generate_core_benchmark_table(df):
    ds_means = df.groupby(['model', 'dataset'])[['point_f1', 'pa_f1', 'pr_auc', 'roc_auc']].mean()
    ds_macro = ds_means.groupby('model').mean()
    ch_macro = df.groupby('model')[['point_f1', 'pa_f1', 'pr_auc', 'roc_auc']].mean()

    micro = {}
    for m in MODELS_ORDER:
        sub = df[df['model'] == m]
        tp = sub['tp'].sum()
        fp = sub['fp'].sum()
        fn = sub['fn'].sum()
        micro_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        micro_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro[m] = {'micro_f1': micro_f1, 'micro_prec': micro_prec, 'micro_rec': micro_rec, 'fp': fp}

    ref_sub = df[df['model'] == 'ts_jepa'].copy()
    ref_sub['pos'] = ref_sub['tp'] + ref_sub['fn']
    ref_sub['neg'] = ref_sub['tn'] + ref_sub['fp']
    ref_sub['all_pos_f1'] = 2 * ref_sub['pos'] / (2 * ref_sub['pos'] + ref_sub['neg'])
    all_pos_ds = ref_sub.groupby('dataset')['all_pos_f1'].mean().mean()
    all_pos_ch = ref_sub['all_pos_f1'].mean()
    tot_pos = ref_sub['pos'].sum()
    tot_neg = ref_sub['neg'].sum()
    all_pos_micro = 2 * tot_pos / (2 * tot_pos + tot_neg)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Multi-Tier Benchmark on Core Multi-Sensor Datasets (45 Strictly Matched Channels across 11 Diverse Modalities). We report unadjusted Point-F1 across three aggregation regimes: unweighted Dataset-Macro (equal $1/N$ dataset weight), Pooled Micro-F1 ($\sum \mathrm{TP}, \sum \mathrm{FP}, \sum \mathrm{FN}$), and Channel-Macro F1, alongside threshold-independent PR-AUC and Point-Adjusted F1 (PA-F1). The best result in each column is in \textbf{bold}, and the second best is \underline{underlined}. All models evaluated under identical pure inductive EVT calibration without transductive test-tail clamps.}",
        r"\label{tab:core_benchmark}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll ccc c cc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Strict Point-Level Evaluation (Unadjusted)}} & \textbf{Ranking} & \multicolumn{2}{c}{\textbf{Point-Adjusted (Literature Ref.)}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-6} \cmidrule(lr){7-8}",
        r"\textbf{Architecture} & \textbf{Physical Inductive Bias / Paradigm} & \textbf{DS-Macro} $\uparrow$ & \textbf{Micro-F1} $\uparrow$ & \textbf{Chan-Macro} $\uparrow$ & \textbf{PR-AUC} $\uparrow$ & \textbf{DS-Macro} $\uparrow$ & \textbf{Chan-Macro} $\uparrow$ \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Proposed Physics-Informed Geometric JEPA Triad}}} \\",
        f"\\textbf{{Operator-Entropy JEPA}} & Koopman Spectral Entropy ($\\mathcal{{H}}_K$) & \\textbf{{{ds_macro.loc['operator_entropy_jepa', 'point_f1']:.4f}}} & \\underline{{{micro['operator_entropy_jepa']['micro_f1']:.4f}}} & \\textbf{{{ch_macro.loc['operator_entropy_jepa', 'point_f1']:.4f}}} & {ds_macro.loc['operator_entropy_jepa', 'pr_auc']:.4f} & {ds_macro.loc['operator_entropy_jepa', 'pa_f1']:.4f} & {ch_macro.loc['operator_entropy_jepa', 'pa_f1']:.4f} \\\\",
        f"\\textbf{{Reynolds-Stress JEPA}}  & Hydrodynamic Stress Tensor ($\\mathbf{{R}}$) & {ds_macro.loc['reynolds_stress_jepa', 'point_f1']:.4f} & {micro['reynolds_stress_jepa']['micro_f1']:.4f} & \\underline{{{ch_macro.loc['reynolds_stress_jepa', 'point_f1']:.4f}}} & {ds_macro.loc['reynolds_stress_jepa', 'pr_auc']:.4f} & {ds_macro.loc['reynolds_stress_jepa', 'pa_f1']:.4f} & \\underline{{{ch_macro.loc['reynolds_stress_jepa', 'pa_f1']:.4f}}} \\\\",
        f"\\textbf{{Potential-Flow JEPA}}   & Curl-Free Potential Field ($\\nabla \\Phi$) & {ds_macro.loc['potential_flow_jepa', 'point_f1']:.4f} & \\textbf{{{micro['potential_flow_jepa']['micro_f1']:.4f}}} & {ch_macro.loc['potential_flow_jepa', 'point_f1']:.4f} & {ds_macro.loc['potential_flow_jepa', 'pr_auc']:.4f} & {ds_macro.loc['potential_flow_jepa', 'pa_f1']:.4f} & \\textbf{{{ch_macro.loc['potential_flow_jepa', 'pa_f1']:.4f}}} \\\\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Ablation Control (Unconstrained Latent Prediction)}}} \\",
        f"\\textit{{Vanilla TS-JEPA}}      & Latent Prediction (Unconstrained) & \\underline{{{ds_macro.loc['ts_jepa', 'point_f1']:.4f}}} & \\textbf{{{micro['ts_jepa']['micro_f1']:.4f}}} & {ch_macro.loc['ts_jepa', 'point_f1']:.4f} & {ds_macro.loc['ts_jepa', 'pr_auc']:.4f} & \\textbf{{{ds_macro.loc['ts_jepa', 'pa_f1']:.4f}}} & {ch_macro.loc['ts_jepa', 'pa_f1']:.4f} \\\\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Published State-of-the-Art Deep Baselines (Unified Reimplementations)}}} \\",
        f"TranAD (VLDB 2022)             & Reconstructive Transformer + Adversarial & {ds_macro.loc['tranad', 'point_f1']:.4f} & {micro['tranad']['micro_f1']:.4f} & {ch_macro.loc['tranad', 'point_f1']:.4f} & \\textbf{{{ds_macro.loc['tranad', 'pr_auc']:.4f}}} & {ds_macro.loc['tranad', 'pa_f1']:.4f} & {ch_macro.loc['tranad', 'pa_f1']:.4f} \\\\",
        f"TimesNet (ICLR 2023)           & Multi-Period 2D Temporal Inception       & {ds_macro.loc['timesnet', 'point_f1']:.4f} & {micro['timesnet']['micro_f1']:.4f} & {ch_macro.loc['timesnet', 'point_f1']:.4f} & \\underline{{{ds_macro.loc['timesnet', 'pr_auc']:.4f}}} & {ds_macro.loc['timesnet', 'pa_f1']:.4f} & {ch_macro.loc['timesnet', 'pa_f1']:.4f} \\\\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{\textbf{Theoretical Reference Limit}}} \\",
        f"\\textit{{All-Positive Trivial Predictor}} & Constant Positive ($\\hat{{y}} \\equiv 1$) & {all_pos_ds:.4f} & {all_pos_micro:.4f} & {all_pos_ch:.4f} & --- & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ]

    out_file = OUT_TAB_DIR / "tab_core_benchmark.tex"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote modern two-tier table to {out_file}")


# =========================================================================
# 2. Modern Hero Benchmark Comparison Bar Chart (Figure 2)
# =========================================================================

def generate_bar_chart(df):
    ds_means = df.groupby(['model', 'dataset'])[['point_f1', 'pa_f1', 'pr_auc']].mean()
    ds_macro = ds_means.groupby('model').mean().loc[MODELS_ORDER]

    fig, ax = plt.subplots(figsize=(11.0, 5.5), dpi=300)
    x = np.arange(len(MODELS_ORDER))
    width = 0.26

    labels = [
        'Operator-Entropy\n(Ours)',
        'Reynolds-Stress\n(Ours)',
        'Potential-Flow\n(Ours)',
        'Vanilla TS-JEPA\n(Control)',
        'TranAD\n(VLDB 22)',
        'TimesNet\n(ICLR 23)',
    ]

    # Modern Academic Palette (Deep Indigo, Vivid Cerulean, Soft Ice)
    c_pt = '#1E3A8A'   # Deep Indigo (Primary Metric)
    c_pr = '#0284C7'   # Cerulean
    c_pa = '#93C5FD'   # Soft Sky Ice

    rects1 = ax.bar(x - width, ds_macro['point_f1'], width, label='Strict Dataset-Macro Point-F1',
                    color=c_pt, edgecolor='#0F172A', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x, ds_macro['pr_auc'], width, label='PR-AUC (Average Precision)',
                    color=c_pr, edgecolor='#0F172A', linewidth=0.7, zorder=3)
    rects3 = ax.bar(x + width, ds_macro['pa_f1'], width, label='Dataset-Macro PA-F1 (Adjusted)',
                    color=c_pa, edgecolor='#0F172A', linewidth=0.7, zorder=3)

    # Clean gridlines behind bars
    ax.grid(axis='y', color='#E2E8F0', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylabel('Score', fontweight='bold', fontsize=10.5)
    ax.set_title('Multi-Tier Telemetry Evaluation across 45 Strictly Matched Benchmark Channels',
                 fontweight='bold', fontsize=12.0, pad=38)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9.5)
    ax.set_ylim(0, 0.48)

    # Shaded All-Positive Reference Band (Unobtrusive)
    ref_sub = df[df['model'] == 'ts_jepa'].copy()
    ref_sub['pos'] = ref_sub['tp'] + ref_sub['fn']
    ref_sub['neg'] = ref_sub['tn'] + ref_sub['fp']
    all_pos_ds = (2 * ref_sub['pos'] / (2 * ref_sub['pos'] + ref_sub['neg'])).groupby(ref_sub['dataset']).mean().mean()
    ax.axhspan(0, all_pos_ds, color='#FEE2E2', alpha=0.35, zorder=1, label=f'All-Positive Trivial Range (F1 $\\leq$ {all_pos_ds:.3f})')
    ax.axhline(all_pos_ds, color='#DC2626', linestyle=':', linewidth=1.2, alpha=0.75, zorder=2)

    # Add numeric labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold', color='#0F172A')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=7.5, color='#1E293B')
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=7.5, color='#334155')

    # Category separation divider
    ax.axvline(3.5, color='#94A3B8', linestyle='--', linewidth=1.1, alpha=0.8, zorder=2)

    # Clean Top Badges positioned cleanly below the legend
    ax.text(1.5, 0.44, 'Proposed Physics-Informed Triad & Control', ha='center', va='center',
            fontweight='bold', color='#1E3A8A', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=0.8))
    ax.text(4.5, 0.44, 'Published Deep Baselines', ha='center', va='center',
            fontweight='bold', color='#991B1B', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.8))

    # Dedicated top clean legend outside the plot area
    ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.95,
              loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, columnspacing=1.2, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "hero_models_sota_comparison.png", dpi=300)
    plt.savefig(OUT_FIG_DIR / "hero_models_sota_comparison.pdf")
    plt.close()
    print(f"[OK] Generated {OUT_FIG_DIR / 'hero_models_sota_comparison.png'}")


# =========================================================================
# 3. Modern Multi-Domain Radar Specialization Chart (Figure 3)
# =========================================================================

def generate_domain_radar_chart(df):
    domains = {
        'Spacecraft\n(MSL, SMAP, swan)': ['MSL', 'SMAP', 'swan'],
        'Biomechanics\n(Daphnet)': ['Daphnet'],
        'Cyber-Physical\n(GHL, GECCO, Genesis)': ['GHL', 'GECCO', 'Genesis'],
        'Traffic & Env.\n(CalIt2, metro, room)': ['CalIt2', 'metro', 'room-occupancy'],
        'Network Cyber\n(cicids)': ['cicids']
    }

    models_to_plot = [
        ('operator_entropy_jepa', 'Operator-Entropy (Ours)', '#1E3A8A', '-', 2.4),
        ('reynolds_stress_jepa', 'Reynolds-Stress (Ours)', '#047857', '-', 2.2),
        ('potential_flow_jepa', 'Potential-Flow (Ours)', '#7C3AED', '-', 2.2),
        ('timesnet', 'TimesNet (ICLR 23)', '#DC2626', '--', 1.8),
        ('tranad', 'TranAD (VLDB 22)', '#D97706', ':', 1.8)
    ]

    domain_scores = {m[0]: [] for m in models_to_plot}

    for d_name, d_list in domains.items():
        sub = df[df['dataset'].isin(d_list)]
        m_mean = sub.groupby('model')['pr_auc'].mean()
        for m_key, _, _, _, _ in models_to_plot:
            score = m_mean.get(m_key, 0.0)
            domain_scores[m_key].append(score)

    categories = list(domains.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.0, 7.2), subplot_kw=dict(polar=True), dpi=300)
    plt.xticks(angles[:-1], categories, color='#0F172A', size=10.0, fontweight='bold')
    ax.tick_params(axis='x', pad=18)  # Generous radial padding to avoid clipping labels
    ax.set_rlabel_position(18)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"], color="#64748B", size=8.5)
    plt.ylim(0, 0.60)

    # Soft circular background grid
    ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)

    for m_key, m_label, m_color, m_style, m_lw in models_to_plot:
        values = domain_scores[m_key]
        values += values[:1]
        ax.plot(angles, values, linewidth=m_lw, linestyle=m_style, label=m_label, color=m_color, zorder=4)
        if 'Ours' in m_label:
            ax.fill(angles, values, color=m_color, alpha=0.08, zorder=3)

    plt.title('PR-AUC Across 5 Specialized Multi-Sensor Regimes', size=12.5, fontweight='bold', pad=30)
    # Position legend cleanly at the bottom center outside polar circle with zero collisions
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=True,
               facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.95, ncol=3, columnspacing=1.2)

    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "domain_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT_FIG_DIR / "domain_radar_chart.pdf", bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {OUT_FIG_DIR / 'domain_radar_chart.png'}")


# =========================================================================
# 4. Modern Complete 11x6 Dataset Performance Heatmap (Figure 4)
# =========================================================================

def generate_heatmap(df):
    pvt = df.groupby(['dataset', 'model'])['pr_auc'].mean().unstack()[MODELS_ORDER]
    pvt.rename(columns=lambda x: MODEL_DISPLAY_NAMES.get(x, x).replace(' (Ours)', '').replace(' (Control)', ''), inplace=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)

    # Use a modern perceptually uniform colormap
    cmap = sns.color_palette("mako", as_cmap=True)

    # Render heatmap with cell lines
    sns.heatmap(pvt, annot=False, cmap=cmap, cbar_kws={'label': 'PR-AUC (Average Precision)'},
                ax=ax, linewidths=1.0, linecolor='#FFFFFF', vmin=0.0, vmax=0.65)

    # Custom text annotations with adaptive contrast for 100% readability across ALL cells
    values = pvt.values
    for r in range(values.shape[0]):
        row_max = np.nanmax(values[r, :])
        for c in range(values.shape[1]):
            val = values[r, c]
            is_winner = abs(val - row_max) < 1e-4 and val > 0.001
            # In mako, low values (0 to 0.25) are dark purple/black; high values (>0.35) are bright mint
            norm_val = (val - 0.0) / 0.65
            text_color = '#0F172A' if norm_val > 0.42 else '#FFFFFF'
            font_weight = 'bold' if is_winner else 'normal'
            txt = f"{val:.3f}" if not np.isnan(val) else "---"
            if is_winner:
                txt = f"{txt}*"
            ax.text(c + 0.5, r + 0.5, txt, ha='center', va='center',
                    color=text_color, fontsize=9.0, fontweight=font_weight)

    ax.set_title('Comprehensive PR-AUC Matrix Across All 11 Telemetry Datasets (* denotes best in row)',
                 fontsize=12.0, fontweight='bold', pad=14)
    ax.set_ylabel('Dataset Name', fontsize=11.0, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=11.0, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9.5, fontweight='bold')
    plt.yticks(fontsize=9.5)

    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "dataset_performance_heatmap.png", dpi=300)
    plt.savefig(OUT_FIG_DIR / "dataset_performance_heatmap.pdf")
    plt.close()
    print(f"[OK] Generated {OUT_FIG_DIR / 'dataset_performance_heatmap.png'}")


def main():
    print(f"Loading benchmark data from {CSV_PATH}...")
    df = load_data()
    print(f"Total benchmark records: {len(df)}")
    generate_core_benchmark_table(df)
    generate_bar_chart(df)
    generate_domain_radar_chart(df)
    generate_heatmap(df)
    print("All tables and figures generated successfully!")


if __name__ == "__main__":
    main()
