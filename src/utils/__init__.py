"""Utilities package for NCAD-CS.

Core scoring and calibration logic has moved to :mod:`src.scoring`.
This package retains only generic utilities: logging setup and plotting.

Plotting helpers are imported lazily so that core modules do not require
matplotlib/Pillow at import time.
"""

from src.utils.logging_utils import setup_logging

__all__ = [
    "setup_logging",
    "plot_channel_diagnostics",
]


def __getattr__(name: str):
    """Lazy import for plotting helpers to avoid eager matplotlib dependency."""
    if name == "plot_channel_diagnostics":
        from src.utils.plotting import plot_channel_diagnostics

        return plot_channel_diagnostics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
