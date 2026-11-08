# 19-Stream Circuit-Breaker Studies (Cohorts 2-4)

Reproduction scripts for the panel ablations, quiet-fault suite, and dyadic front-end ablation.

> **Development-set notice.** The panel's calibration constants (percentiles, multipliers, floors) were
> selected while inspecting test metrics on 23 streams: the 19 non-GHL primary streams in
> `reports/streams_19_nonghl.txt` plus SMAP T-2, MSL P-10, Daphnet S09R01E2, and OPPORTUNITY S1-ADL3
> (`common.DEVELOPMENT_STREAMS`). Results from `run_perspective_ladder.py` and
> `run_component_ablations.py` are therefore development-set results. The held-out test of the frozen
> panel is the 118 streams of the 141-stream benchmark that are not in this set.

| Script | Cohort / table | Training | Output |
|---|---|---|---|
| `build_cache.py` | trains models for Cohort 2 | Soliton-Reynolds JEPA, 15 epochs, batch 16, constant LR, EMA 0.996, seeds 42/123/456 | `reports/cache_19stream_panel/*.npz` |
| `run_perspective_ladder.py` | Cohort 2, Table II(a) | uses cache (CPU only) | `reports/perspective_ladder_19stream.csv` |
| `run_component_ablations.py` | Cohort 2, Table II(b) | uses cache (CPU only) | `reports/ablation_224bit_components_19stream.csv` |
| `run_quiet_fault_evaluation.py` | Cohort 3, Fig. 3(b) | 6 streams, seed 42, 10 epochs, batch 32, clip 1.0, EMA 0.995 | `reports/quiet_fault_224bit_evaluation.csv` |
| `run_dyadic_ablation.py` | Cohort 4, Sec. VI-C | 5 streams x 3 seeds, 12 epochs, batch 32, clip 1.0, EMA 0.995 | `reports/dyadic_frontend_224bit_ablation.csv` |

Run from the repository root with the `alpha` environment, e.g.

```bash
python -m scripts.circuit_breaker_19stream.build_cache
python -m scripts.circuit_breaker_19stream.run_perspective_ladder
python -m scripts.circuit_breaker_19stream.run_component_ablations
```

Set `NCAD_PANEL_CACHE=<dir>` to reuse an existing residual cache.

**Known protocol differences from the 141-stream benchmark:** the cache stores whitened residuals only,
so Cohort 2 omits the calibration-stabilized blend of Eq. (3); Cohorts 3-4 apply it.
