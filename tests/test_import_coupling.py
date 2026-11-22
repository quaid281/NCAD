"""Regression tests for package import coupling.

Core numerical utilities must not eagerly import matplotlib/Pillow so that
headless and test environments without plotting dependencies remain functional.
"""

import importlib
import sys


def _reload_without_plotting(module_name: str):
    """Import a module fresh and assert matplotlib/PIL were not pulled in."""
    for mod in list(sys.modules):
        if mod.startswith("src.utils") or mod.startswith("src.engine") or mod in ("matplotlib", "PIL"):
            del sys.modules[mod]
    importlib.import_module(module_name)
    return "matplotlib" in sys.modules, "PIL" in sys.modules


def test_evt_calibrator_does_not_import_matplotlib():
    mpl, pil = _reload_without_plotting("src.scoring.evt_calibrator")
    assert not mpl, "src.scoring.evt_calibrator should not eagerly import matplotlib"
    assert not pil, "src.scoring.evt_calibrator should not eagerly import PIL"


def test_trainer_does_not_import_matplotlib():
    mpl, pil = _reload_without_plotting("src.engine.trainer")
    assert not mpl, "src.engine.trainer should not eagerly import matplotlib"
    assert not pil, "src.engine.trainer should not eagerly import PIL"


def test_evaluator_does_not_import_matplotlib():
    mpl, pil = _reload_without_plotting("src.engine.evaluator")
    assert not mpl, "src.engine.evaluator should not eagerly import matplotlib"


def test_plotting_lazy_attribute():
    """plot_channel_diagnostics remains accessible via lazy import."""
    from src.utils import plot_channel_diagnostics

    assert callable(plot_channel_diagnostics)
