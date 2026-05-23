# NCAD-CS v4: Counterfactual Successor Memory

This directory is an isolated prototype for the NCAD-CS direction. It does not replace `notebooks_v3`.

The accepted v4 configuration uses the original hybrid-pooling TCN encoder with Counterfactual Successor Memory and the adaptive-elbow scorer. v4 also owns an experimental multi-scale TCN fork, but it is opt-in through `--encoder multi_scale_tcn` because the all-channel benchmark did not improve enough to make it the default.

The central change is that v4 stops substituting the context window for scoring. Instead, it stores normal contexts together with the normal suspect segment that followed them. During inference, nearest normal contexts retrieve plausible normal successors, and the observed suspect segment is scored against those counterfactual normal futures.

## Main Idea

```text
Given this context, what should the next 16 samples have looked like?
```

v4 keeps evidence streams separate before event fusion:

- `successor_score`: observed suspect segment vs retrieved normal successors.
- `manifold_uncertainty_score`: successor residual normalized by neighbor-successor dispersion, so deviations from a tight normal successor set count more than deviations from a diffuse successor set.
- `context_ratio`: current context distance relative to normal context-memory distances.
- `local_score`: abrupt point or mean-shift deviation inside the suspect segment.
- `window_score`: fused event evidence used for point-level predictions.

By default, v4 uses the prior adaptive-elbow counterfactual successor scorer. Successor Manifold Uncertainty is retained as an experimental option behind `--manifold-uncertainty`. In that mode, the raw successor residual is down-weighted when retrieved normal successors disagree, while a calibrated manifold-normalized residual is retained as high-confidence evidence when the successor set is tight. This targets two observed failure modes: false-positive collapse on diffuse successor neighborhoods and conservative under-detection when the retrieved normal futures agree strongly.

The final threshold uses the training normal successor distribution plus a channel-adaptive score-distribution floor from the test channel. The default floor is selected from the channel's score geometry using an elbow consensus over the sorted score curve and histogram separation. This prevents near-zero training residuals on flat channels from turning into all-positive predictions without requiring a fixed percentile. If the score distribution contains a clipped maximum-score plateau, v4 first estimates the elbow below that plateau so saturated anomaly evidence is not hidden by an equal threshold.

## Run a Probe

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --channels A-1 D-3 P-3 T-5 --epochs 5 --output-dir notebooks_v4/results/probe_e5
```

For a faster smoke test:

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --channels A-1 D-3 --epochs 1 --max-train-windows 512 --max-test-windows 1024 --max-memory-windows 512 --no-plots --output-dir notebooks_v4/results/smoke
```

For sustained channels such as `D-3` and `P-3`, the adaptive elbow floor is now the default:

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --channels D-3 P-3 --epochs 3 --max-train-windows 1024 --max-memory-windows 1024 --output-dir notebooks_v4/results/probe_e3_adaptive_d3_p3
```

`--score-floor-percentile` is retained only as a legacy override for reproducing older percentile-floor runs.

To run the experimental manifold-uncertainty variant explicitly:

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --all --epochs 15 --max-memory-windows 5000 --no-plots --manifold-uncertainty --output-dir notebooks_v4/results/all_channels_e15_manifold_uncertainty
```

To run the experimental multi-scale encoder explicitly:

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --all --epochs 15 --max-memory-windows 5000 --no-plots --encoder multi_scale_tcn --output-dir notebooks_v4/results/all_channels_e15_multiscale
```

## Compare With v3

Run v3 and v4 into separate result folders, then compare their `summary.csv` files:

```powershell
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v3/train.py --channels A-1 D-3 P-3 T-5 --epochs 5 --output-dir notebooks_v3/results/compare_probe_e5
C:/Users/andre/anaconda3/envs/pinknoise/python.exe notebooks_v4/train.py --channels A-1 D-3 P-3 T-5 --epochs 5 --output-dir notebooks_v4/results/compare_probe_e5
```

Each v4 channel folder saves:

- `metrics.json`
- `point_predictions.csv`
- `window_scores.csv`
- `successor_memory.npz`
- `encoder.pt`
- `feature_metadata.json`
- a diagnostic PNG unless `--no-plots` is used