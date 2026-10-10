"""Build the definitive machine-readable stream manifest (Reviewer reproducibility #1).

Scans mTSBench_data/ for every stream listed in reports/streams_45.txt and records
channels, split lengths, anomaly counts, benchmark role, and the seeds actually
evaluated per stream (derived from the held-out benchmark CSVs themselves, so the
manifest can never disagree with the reported results).

Output: reports/stream_manifest.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "mTSBench_data"
REPORTS = ROOT / "reports"

LABEL_COLS = {"is_anomaly", "label", "labels"}
DROP_COLS = {"timestamp"} | LABEL_COLS


def stream_files(ds: str, ch: str):
    d = DATA / ds
    if ch == "default":
        tr = d / f"{ds}_train.csv"
        tr = tr if tr.exists() else d / f"{ds}.csv"
        te = d / f"{ds}_test.csv"
    else:
        tr = d / f"{ds}_{ch}_train.csv"
        tr = tr if tr.exists() else d / f"{ch}_train.csv"
        te = d / f"{ds}_{ch}_test.csv"
        te = te if te.exists() else d / f"{ch}_test.csv"
    return tr, te


def main():
    primary = set()
    for m in ["streams_19_nonghl.txt", "streams_14_ghl.txt"]:
        p = REPORTS / m
        if p.exists():
            primary |= {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}

    held = []
    for f in ["heldout_45stream.csv", "heldout_45stream_b.csv", "heldout_45stream_c.csv",
              "heldout_nonghl_seeds45.csv", "heldout_nonghl_seeds45_a.csv",
              "heldout_nonghl_seeds45_b.csv",
              "heldout_ghl_seeds45_a.csv", "heldout_ghl_seeds45_b.csv"]:
        p = REPORTS / f
        if p.exists():
            held.append(pd.read_csv(p))
    held = pd.concat(held, ignore_index=True)
    held = held[(held["calibrator"] == "SPOT") & (held["mapping"] == "smear")]
    seeds = held.groupby(["dataset", "channel"])["seed"].apply(
        lambda s: sorted(s.unique())).to_dict()

    rows = []
    streams = [ln.strip() for ln in (REPORTS / "streams_45.txt").read_text().splitlines()
               if ln.strip()]
    for spec in streams:
        ds, ch = spec.split(":", 1)
        tr, te = stream_files(ds, ch)
        tag = f"{ds}:{ch}"
        if not tr.exists() or not te.exists():
            rows.append({"dataset": ds, "stream": ch, "status": "MISSING_FILES",
                         "stream_id": tag})
            print(f"{tag}: MISSING ({tr.exists()}, {te.exists()})")
            continue
        a = pd.read_csv(tr)
        b = pd.read_csv(te)
        k = len([c for c in a.columns if c not in DROP_COLS])
        lc = [c for c in b.columns if c in LABEL_COLS]
        y = b[lc[0]].to_numpy().astype(int) if lc else np.zeros(len(b), dtype=int)
        sd = seeds.get((ds, ch), [])
        rows.append({
            "dataset": ds,
            "stream": ch,
            "stream_id": tag,
            "n_channels": k,
            "n_train": len(a),
            "n_test": len(b),
            "n_anomalous_test": int(y.sum()),
            "anomaly_rate_test": round(float(y.mean()), 6),
            "role": "primary_benchmark" if tag in primary else "case_study",
            "seeds_evaluated": ";".join(map(str, sd)),
            "n_seeds": len(sd),
            "split": "65/17.5/17.5 train/cal/eval + 64-step isolation gap",
            "status": "ok",
        })
        print(f"{tag}: K={k} train={len(a)} test={len(b)} anom={int(y.sum())} "
              f"seeds={sd} role={'primary' if tag in primary else 'case'}")

    out = pd.DataFrame(rows)
    path = REPORTS / "stream_manifest.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path} ({len(out)} streams, "
          f"{int((out.role == 'primary_benchmark').sum())} primary)")


if __name__ == "__main__":
    main()
