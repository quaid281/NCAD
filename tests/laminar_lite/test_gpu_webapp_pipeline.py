"""Verification test for the local GPU Laminar-JEPA-Lite interactive explorer pipeline."""

from __future__ import annotations

from app.gpu_pipeline import list_available_streams, run_laminar_lite_on_gpu


def test_list_available_streams_discovers_datasets():
    catalog = list_available_streams()
    assert "datasets" in catalog
    assert "presets" in catalog
    assert "GECCO" in catalog["datasets"]
    assert "GECCO_water_quality" in catalog["datasets"]["GECCO"]


def test_run_laminar_lite_on_gpu_gecco_smoke():
    payload = run_laminar_lite_on_gpu(
        dataset="GECCO",
        stream="GECCO_water_quality",
        epochs=2,
        seed=42,
        max_test_steps=900,
    )
    meta = payload["meta"]
    assert meta["n_channels"] == 9
    assert meta["n_frames"] > 50
    assert len(payload["latent"]["barycenter_2d"]) == 2
    assert len(payload["latent"]["te_pred_2d"]) == 3
    assert len(payload["latent"]["te_tgt_2d"]) == 3
    assert len(payload["whitening"]["live_zca_2d"]) == meta["n_frames"]
    assert len(payload["dials"]["ratios"][0]) == 128
    assert payload["thresholds"]["k_lite"] >= 3.5
