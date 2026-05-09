# Dataset Inventory Summary

Generated: 2026-05-09T09:14:38

Usable channel-level datasets: 82
Spacecraft/channel groups: {'SMAP': 55, 'MSL': 27}
Total training samples across usable channels: 199,142
Total test samples across usable channels: 517,764
Total labeled anomalous test points: 64,704
Overall labeled anomaly rate: 12.50%

Suggested manuscript framing:

Although the telemetry comes from a public spacecraft benchmark, each channel is an independent univariate anomaly-detection dataset with its own train split, test split, operating regime, and anomaly annotations. The experimental suite therefore evaluates NCAD-CS over 82 channel-level datasets rather than a single time series.
