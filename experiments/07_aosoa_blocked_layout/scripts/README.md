# Experiment 07 Scripts

## Purpose
Utilities for collecting, aggregating, and plotting Experiment 07 AoSoA blocked-layout runs.

## Scripts
- `collect_run.py`: copy latest benchmark JSON into `runs/<gpu>/timestamp[_label].json`
- `analyze_aosoa_blocked_layout.py`: aggregate multi-run data into CSV tables and charts
- `plot_results.py`: quick single-run plot from `results/tables/benchmark_results.json`

## Typical Workflow
Run benchmark and collect:
```powershell
python scripts/run_experiment_data_collection.py --experiment 07_aosoa_blocked_layout --iterations 10 --warmup 3 --size 512M
```

Regenerate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 07_aosoa_blocked_layout --collect-run
```

Direct analysis call:
```powershell
python experiments/07_aosoa_blocked_layout/scripts/analyze_aosoa_blocked_layout.py
```
