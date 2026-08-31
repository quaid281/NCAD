"""Command-line interface and argument parser for NCAD-CS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CSMConfig
from src.data.data_loader import DataLoader
from src.engine.orchestrator import run_experiment
from src.models.registry import canonical_model_choices
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI options for NCAD-CS."""
    parser = argparse.ArgumentParser(description="Run NCAD-CS Counterfactual Successor Memory.")
    parser.add_argument("--channel", type=str, default=None, help="Run one channel, for example A-1.")
    parser.add_argument("--channels", type=str, nargs="+", default=None, help="Run selected channels.")
    parser.add_argument("--all", action="store_true", help="Run every channel with train and test files.")
    parser.add_argument("--list-channels", action="store_true", help="List available channels and exit.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the data directory. Defaults to ../data.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for run outputs.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["hybrid_tcn", "multi_scale_tcn"],
        default="hybrid_tcn",
        help="Encoder architecture.",
    )
    parser.add_argument("--successor-neighbors", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for patch_ts_jepa model.")
    parser.add_argument("--event-threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--score-floor-percentile",
        type=float,
        default=None,
        help="Legacy fixed percentile floor override. Omit to use the adaptive elbow floor.",
    )
    parser.add_argument(
        "--manifold-uncertainty",
        action="store_true",
        help="Enable the experimental successor manifold uncertainty scorer.",
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--max-memory-windows", type=int, default=5000)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(canonical_model_choices()),
        default="ts_jepa",
        help="Core model type (ts_jepa, patch_ts_jepa, gat_jepa, or legacy ncad).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Structured logger level.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    loader = DataLoader(args.data_dir)
    available_channels = loader.list_channels()
    if args.list_channels:
        print("Available channels:")
        for channel in available_channels:
            print(f"  {channel}")
        return
    if args.channel:
        channels = [args.channel]
    elif args.channels:
        channels = args.channels
    else:
        channels = available_channels if args.all else ["A-1"]
    missing = [channel for channel in channels if channel not in available_channels]
    if missing:
        raise ValueError(f"Unknown or incomplete channels: {missing}")

    config = CSMConfig(
        model_type=args.model_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        encoder_architecture=args.encoder,
        successor_neighbors=args.successor_neighbors,
        patch_size=args.patch_size,
        event_threshold_percentile=args.event_threshold_percentile,
        score_floor_percentile=args.score_floor_percentile,
        manifold_uncertainty=args.manifold_uncertainty,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        max_memory_windows=args.max_memory_windows,
        save_plots=not args.no_plots,
        device=args.device,
    )
    run_dir, summary = run_experiment(channels, config)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
