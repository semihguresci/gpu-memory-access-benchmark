# Experiment 13 Scripts

## Purpose
These scripts collect, analyze, and plot Experiment 13 (`13_scatter_access_pattern`) benchmark outputs.

## Typical Flow
1. Run the benchmark binary and write `results/tables/benchmark_results.json`.
2. Archive that JSON with `collect_run.py`.
3. Generate CSV tables with `analyze_scatter_access_pattern.py`.
4. Generate charts with `plot_results.py`.

## Commands
```bash
python experiments/13_scatter_access_pattern/scripts/collect_run.py
python experiments/13_scatter_access_pattern/scripts/analyze_scatter_access_pattern.py
python experiments/13_scatter_access_pattern/scripts/plot_results.py
```

## Outputs
- `results/tables/scatter_access_pattern_summary.csv`
- `results/tables/scatter_access_pattern_relative.csv`
- `results/tables/scatter_access_pattern_stability.csv`
- `results/tables/scatter_access_pattern_contention.csv`
- `results/charts/scatter_access_pattern_*.png`
- `runs/<gpu>/<timestamp>.json`
