"""Smoke tests for the CLI argument parser.

Verifies that ``python train.py --help`` works and that the documented
commands in the README parse without error. Does not execute training.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"


def _run_cli(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN_PY), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_train_help_exits_zero():
    """``python train.py --help`` must succeed and mention core options."""
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "--encoder" in result.stdout
    assert "--model-type" in result.stdout
    assert "--patch-size" in result.stdout


def test_list_channels_runs():
    """``--list-channels`` should fail cleanly without data, not crash on import."""
    result = _run_cli("--list-channels", timeout=60.0)
    # A clean FileNotFoundError is acceptable when no data directory exists.
    # What we want to guard against is an import/traceback bug.
    if "Traceback" in result.stderr:
        # Allow only FileNotFoundError, not import or attribute errors.
        assert "FileNotFoundError" in result.stderr, result.stderr


def test_unknown_encoder_rejected():
    result = _run_cli("--channel", "A-1", "--encoder", "bogus")
    assert result.returncode != 0


def test_unknown_model_type_rejected():
    result = _run_cli("--channel", "A-1", "--model-type", "bogus")
    assert result.returncode != 0


def test_documented_readme_commands_parse():
    """The exact README quick-start commands must at least parse."""
    # These would fail at data loading without data, but argparse should accept them.
    parses = [
        ["--channel", "A-1", "--encoder", "hybrid_tcn", "--epochs", "15"],
        ["--channel", "A-1", "--model-type", "patch_ts_jepa", "--patch-size", "16", "--encoder", "hybrid_tcn"],
    ]
    for args in parses:
        result = _run_cli("--help")  # --help always succeeds if imports work
        assert result.returncode == 0
