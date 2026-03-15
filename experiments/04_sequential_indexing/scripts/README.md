# Experiment 04 Scripts

## Purpose
Utilities for collecting, aggregating, and plotting Experiment 04 sequential indexing runs.

## Scripts
- `collect_run.py`: copy latest benchmark JSON into `runs/<gpu>/timestamp[_label].json`
- `analyze_sequential_indexing.py`: aggregate multi-run data into CSV tables and charts
- `plot_results.py`: quick single-run plot from `results/tables/benchmark_results.json`

## Typical Workflow
Run benchmark and collect:
```powershell
python scripts/run_experiment_data_collection.py --experiment 04_sequential_indexing --iterations 10 --warmup 3 --size 64M
```

Regenerate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 04_sequential_indexing --collect-run
```

Direct analysis call:
```powershell
python experiments/04_sequential_indexing/scripts/analyze_sequential_indexing.py
```
