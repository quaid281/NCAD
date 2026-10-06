#!/usr/bin/env python3
"""Verification Pilot: Empirical comparison of Point-level vs Event-level / Range-aware metrics,
dense vs middle mapping, and static vs dynamic EVT thresholding across representative channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_modern_baselines import train_and_score_channel


CHANNELS_TO_TEST = [
    ("room-occupancy", "default"),
    ("SMAP", "P-3"),
    ("CalIt2", "traffic"),
]

MODELS_TO_TEST = [
    "operator_entropy_jepa",
    "potential_flow_jepa",
]

CONFIGURATIONS = [
    {
        "config_name": "Standard (Middle, Smooth=5, Static EVT)",
        "mapping_method": "middle",
        "smoothing_window": 5,
        "dynamic_evt": False,
    },
    {
        "config_name": "Sharp (Dense, Smooth=3, Static EVT)",
        "mapping_method": "dense",
        "smoothing_window": 3,
        "dynamic_evt": False,
    },
    {
        "config_name": "Adaptive (Dense, Smooth=3, Dynamic SPOT)",
        "mapping_method": "dense",
        "smoothing_window": 3,
        "dynamic_evt": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run Incident Metrics Pilot Verification")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--output_csv", type=str, default="reports/pilot_incident_metrics_comparison.csv")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device if args.device != "auto" else "cpu"
    )
    print(f"Running Pilot Verification on device: {device}")

    results = []
    out_path = ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ds_name, chan_name in CHANNELS_TO_TEST:
        ds_dir = ROOT / "mTSBench_data" / ds_name
        if chan_name == "default":
            train_p = ds_dir / f"{ds_name}_train.csv"
            test_p = ds_dir / f"{ds_name}_test.csv"
        else:
            train_p = ds_dir / f"{ds_name}_{chan_name}_train.csv"
            test_p = ds_dir / f"{ds_name}_{chan_name}_test.csv"

        if not train_p.exists() or not test_p.exists():
            print(f"Skipping {ds_name} / {chan_name}: files not found.")
            continue

        for model_name in MODELS_TO_TEST:
            for cfg in CONFIGURATIONS:
                print(
                    f"\n>>> [{model_name}] {ds_name}/{chan_name} | {cfg['config_name']} ... ",
                    end="",
                    flush=True,
                )
                try:
                    res = train_and_score_channel(
                        model_name=model_name,
                        dataset_name=ds_name,
                        chan_name=chan_name,
                        train_path=train_p,
                        test_path=test_p,
                        seed=42,
                        epochs=args.epochs,
                        batch_size=32,
                        mapping_method=cfg["mapping_method"],
                        smoothing_window=cfg["smoothing_window"],
                        dynamic_evt=cfg["dynamic_evt"],
                        device=device,
                    )
                    res["config_name"] = cfg["config_name"]
                    results.append(res)

                    df_inc = pd.DataFrame([res])
                    if not out_path.exists():
                        df_inc.to_csv(out_path, index=False)
                    else:
                        df_inc.to_csv(out_path, mode="a", header=False, index=False)

                    print(
                        f"Pt-F1: {res['point_f1']:.4f} (P: {res['point_precision']:.4f}, R: {res['point_recall']:.4f}, FP: {res['fp']}) | "
                        f"Evt-F1: {res['event_f1']:.4f} (P: {res['event_precision']:.4f}, R: {res['event_recall']:.4f}) | "
                        f"Rng-F1: {res['range_f1']:.4f} (P: {res['range_precision']:.4f}, R: {res['range_recall']:.4f}) | "
                        f"PA-F1: {res['pa_f1']:.4f} ({res['elapsed_sec']}s)"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 90)
        print("SUMMARY TABLE: POINT vs EVENT vs RANGE PERFORMANCE")
        print("=" * 90)
        summary_cols = [
            "dataset",
            "model",
            "config_name",
            "point_f1",
            "point_precision",
            "point_recall",
            "event_f1",
            "event_precision",
            "event_recall",
            "range_f1",
            "fp",
        ]
        print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
