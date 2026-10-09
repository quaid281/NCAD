#!/usr/bin/env python3
"""Evaluate non-neural Linear AR baseline under the held-out calibration protocol."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_multiseed_heldout_benchmark import load_stream
from src.scoring.evt_calibrator import EVTCalibrator
from src.scoring.event_fusion import compute_metrics

STREAMS = [
    ("SMAP", "P-3"), ("MSL", "M-1"), ("SMAP", "A-1"), ("MSL", "C-1"),
    ("SMAP", "A-2"), ("SMAP", "A-7"), ("SMAP", "D-3"), ("SMAP", "E-1"),
    ("MSL", "D-14"), ("MSL", "D-15"),
    ("Daphnet", "S01R01E1"), ("Daphnet", "S01R02E0"), ("Daphnet", "S02R01E0"),
    ("Daphnet", "S02R02E0"), ("Daphnet", "S03R01E0"), ("Daphnet", "S03R02E0"),
    ("Genesis", "default"), ("Metro", "default"), ("SWAN", "sf")
]

def main():
    results = []
    p = 4
    val_frac = 0.35
    gap = 64
    
    for ds, ch in STREAMS:
        try:
            train, test, y = load_stream(ds, ch)
            n_val = int(len(train) * val_frac)
            tr = train[:-n_val]
            val_block = train[-n_val:]
            n_half = len(val_block) // 2
            cal = val_block[:n_half]
            hold = val_block[n_half + gap:]
            
            mu, sd = tr.mean(0), tr.std(0)
            sc = np.where(sd < 1e-8, 1.0, np.maximum(sd, 0.01))
            tr_s = (tr - mu) / sc
            cal_s = (cal - mu) / sc
            hold_s = (hold - mu) / sc
            te_s = (test - mu) / sc
            
            X_tr = np.hstack([tr_s[i:len(tr_s)-p+i] for i in range(p)])
            Y_tr = tr_s[p:]
            reg = Ridge(alpha=1.0).fit(X_tr, Y_tr)
            
            X_cal = np.hstack([cal_s[i:len(cal_s)-p+i] for i in range(p)])
            res_cal = np.linalg.norm(cal_s[p:] - reg.predict(X_cal), axis=1)
            
            X_hold = np.hstack([hold_s[i:len(hold_s)-p+i] for i in range(p)])
            res_hold = np.linalg.norm(hold_s[p:] - reg.predict(X_hold), axis=1)
            
            X_te = np.hstack([te_s[i:len(te_s)-p+i] for i in range(p)])
            res_te = np.linalg.norm(te_s[p:] - reg.predict(X_te), axis=1)
            
            calib = EVTCalibrator(risk_level=1e-3, init_percentile=98.0, adaptive_kurtosis=False)
            calib.fit(res_cal)
            tau = float(calib.threshold_)
            
            hold_fpr = float((res_hold > tau).mean())
            y_aligned = y[p:]
            y_pred = (res_te > tau).astype(int)
            
            m = compute_metrics(y_aligned, y_pred)
            prauc = float(average_precision_score(y_aligned, res_te))
            
            res_dict = {
                "dataset": ds, "channel": ch, "hold_fpr": hold_fpr,
                "point_f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "pr_auc": prauc, "event_recall": m.get("event_recall", 0.0),
                "mean_delay": m.get("mean_delay", 0.0)
            }
            results.append(res_dict)
            print(f"[{ds} {ch}] FPR: {hold_fpr:.4f}, F1: {m['f1']:.4f}, PR-AUC: {prauc:.4f}, Delay: {m.get('mean_delay', 0.0):.1f}")
        except Exception as e:
            print(f"Error on {ds} {ch}: {e}")
            
    df = pd.DataFrame(results)
    out_csv = ROOT / "reports" / "linear_ar_baseline_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== LINEAR AR BASELINE SUMMARY (19 Non-GHL Streams) ===")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    main()
