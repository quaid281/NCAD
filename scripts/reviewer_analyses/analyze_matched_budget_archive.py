"""Analyze the matched-budget archive rerun (reports/reviewer_matched_budget_141_shard*.csv)."""
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = r"c:\Users\andre\OneDrive\Desktop\NCAD_CS"
d = pd.concat([pd.read_csv(f) for f in glob.glob(ROOT + r"\reports\reviewer_matched_budget_141_shard*.csv")])
rng = np.random.default_rng(0)
print("rows", len(d), "streams", d.groupby(["dataset", "channel"]).ngroups, "heldout streams",
      d[d.heldout == True].groupby(["dataset", "channel"]).ngroups)
COLS = ["f1", "event_recall", "hold_fpr", "test_fpr", "mean_delay", "er_f01", "er_f05", "pr_auc"]


def sm(df, model, mapping):
    s = df[(df.model == model) & (df.mapping == mapping)]
    return s.groupby(["dataset", "channel"])[COLS].mean()


def paired(a, b, col):
    j = a[[col]].join(b[[col]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
    x = (j[col + "_a"] - j[col + "_b"]).values
    ds = j.index.get_level_values(0).values
    u = np.unique(ds)
    sb = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    cb = [np.concatenate([x[ds == q] for q in rng.choice(u, len(u))]).mean() for _ in range(4000)]
    nz = x[x != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 5 else np.nan
    # dataset sign-flip on dataset means
    dm = pd.Series(x, index=ds).groupby(level=0).mean().values
    flips = np.array([[1 if (k >> i) & 1 else -1 for i in range(len(dm))] for k in range(2 ** len(dm))])
    obs = abs(dm.mean())
    p_sf = float(np.mean(np.abs((flips * dm).mean(1)) >= obs - 1e-12))
    return dict(n=len(x), d=x.mean(), sci=np.percentile(sb, [2.5, 97.5]).tolist(), cci=np.percentile(cb, [2.5, 97.5]).tolist(),
                wins=int((x > 0).sum()), losses=int((x < 0).sum()), p=p, p_signflip=p_sf, pos_datasets=int((dm > 0).sum()), n_datasets=len(dm))


def fmt(r):
    return (f"d={r['d']:+.4f} streamCI=[{r['sci'][0]:+.4f},{r['sci'][1]:+.4f}] clusterCI=[{r['cci'][0]:+.4f},{r['cci'][1]:+.4f}] "
            f"W/L={r['wins']}/{r['losses']} p={r['p']:.3g} signflip={r['p_signflip']:.3g} ({r['pos_datasets']}/{r['n_datasets']} ds+)")


out = {}
for subset_name, sub in [("heldout118", d[d.heldout == True]), ("all141", d)]:
    print(f"\n######## {subset_name}")
    for mapping in ["trailing", "smear"]:
        print(f"\n==== {mapping} ====")
        models = sorted(sub[sub.mapping == mapping].model.unique())
        T = {m: sm(sub, m, mapping) for m in models}
        for m in models:
            t = T[m]
            f = t.hold_fpr.values
            print(f"{m:24s} n={len(t)} " + " ".join(f"{c}={np.nanmean(t[c]):.4f}" for c in COLS)
                  + f" | FPRmed={np.median(f):.4f} p90={np.percentile(f,90):.4f} <=1%={np.mean(f<=0.01):.3f} <=5%={np.mean(f<=0.05):.3f}")
            out[f"{subset_name}|{mapping}|{m}"] = {c: float(np.nanmean(t[c])) for c in COLS} | {
                "fpr_median": float(np.median(f)), "fpr_p90": float(np.percentile(f, 90)), "fpr_p95": float(np.percentile(f, 95)),
                "share_le1": float(np.mean(f <= 0.01)), "share_le2": float(np.mean(f <= 0.02)), "share_le5": float(np.mean(f <= 0.05)),
                "n_gt1": int((f > 0.01).sum()), "n_gt5": int((f > 0.05).sum()), "n_gt10": int((f > 0.10).sum()), "n": int(len(t))}
        comps = [("Laminar-JEPA-Lite", "Multi-Horizon-JEPA-1D"), ("Laminar-JEPA-Lite", "MH1D-RateMatched-Lite"),
                 ("Lite-RateMatched-1D", "Multi-Horizon-JEPA-1D"), ("Laminar-JEPA-160", "Laminar-JEPA-Unq"),
                 ("Laminar-JEPA-Lite", "Laminar-JEPA-160"), ("Laminar-JEPA-Lite", "Laminar-JEPA-Unq"),
                 ("Laminar-JEPA-160", "Multi-Horizon-JEPA-1D")]
        if mapping == "trailing":
            comps += [("Laminar-JEPA-Lite", b) for b in ["Lite-MaxCoordZ", "Lite-RawTopK", "Lite-ConformalL2", "Lite-Fisher", "Lite-Stouffer", "Lite-BH-Conformal"]]
        for a, b in comps:
            if a not in T or b not in T:
                continue
            print(f"  -- {a} vs {b}")
            for c in ["f1", "event_recall", "hold_fpr", "test_fpr", "mean_delay", "er_f01", "er_f05"]:
                r = paired(T[a], T[b], c)
                out[f"{subset_name}|{mapping}|{a}__vs__{b}|{c}"] = r
                print(f"     {c:12s} {fmt(r)}")
json.dump(out, open(ROOT + r"\reports\reviewer_matched_budget_summary.json", "w"), indent=1, default=float)
print("saved summary json")
