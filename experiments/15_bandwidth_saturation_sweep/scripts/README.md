# Experiment 15 Scripts

## Purpose
These scripts collect, analyze, and plot Experiment 15 (`15_bandwidth_saturation_sweep`) benchmark outputs.

## Typical Flow
1. Run the benchmark binary and write `results/tables/benchmark_results.json`.
2. Archive that JSON with `collect_run.py`.
3. Generate CSV tables with `analyze_bandwidth_saturation_sweep.py`.
4. Generate charts with `plot_results.py`.

## Commands
```bash
python experiments/15_bandwidth_saturation_sweep/scripts/collect_run.py
python experiments/15_bandwidth_saturation_sweep/scripts/analyze_bandwidth_saturation_sweep.py
python experiments/15_bandwidth_saturation_sweep/scripts/plot_results.py
```

## Outputs
- `results/tables/bandwidth_saturation_summary.csv`
- `results/tables/bandwidth_saturation_plateau_summary.csv`
- `results/tables/bandwidth_saturation_status_overview.csv`
- `results/charts/bandwidth_saturation_*.png`
- `runs/<gpu>/<timestamp>.json`
