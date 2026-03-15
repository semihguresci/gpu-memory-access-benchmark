# Experiment 02 Scripts

These scripts are local to Experiment 02 and read/write files under:
- `../results/tables/`
- `../results/charts/`
- `../runs/`

## Usage
From repository root:

```powershell
python experiments/02_local_size_sweep/scripts/collect_run.py
python experiments/02_local_size_sweep/scripts/analyze_local_size_sweep.py
python experiments/02_local_size_sweep/scripts/plot_results.py
```

Root-level convenience scripts:

```powershell
python scripts/run_experiment_data_collection.py --experiment 02_local_size_sweep
python scripts/generate_experiment_artifacts.py --experiment 02_local_size_sweep
```

Typical workflow for multi-device tracking:
1. Run benchmark and write `benchmark_results.json` for the current device.
2. Collect the run into `runs/<device>/...`:
   - `python experiments/02_local_size_sweep/scripts/collect_run.py`
3. Rebuild aggregated tables/charts across all collected runs:
   - `python experiments/02_local_size_sweep/scripts/analyze_local_size_sweep.py`
4. Generate quick summary plot from current JSON:
   - `python experiments/02_local_size_sweep/scripts/plot_results.py`

## Inputs
- `../results/tables/benchmark_results.json` (row-level data required for local-size analysis)
- `../runs/**/*.json` (optional, for multi-device aggregation)

## Outputs
- `../results/tables/local_size_sweep_runs_index.csv`
- `../results/tables/local_size_sweep_multi_run_summary.csv`
- `../results/tables/local_size_sweep_summary.csv`
- `../results/tables/local_size_sweep_best_local_size.csv`
- `../results/tables/local_size_sweep_speedup_vs_ls64.csv`
- `../results/tables/local_size_sweep_local_size_ranking.csv`
- `../results/tables/local_size_sweep_operation_ratio.csv`
- `../results/tables/local_size_sweep_operation_ratio_summary.csv`
- `../results/tables/local_size_sweep_status_overview.csv`
- `../results/tables/local_size_sweep_test_setup.csv`
- `../results/tables/local_size_sweep_gpu_ms_pivot.csv`
- `../results/tables/local_size_sweep_throughput_pivot.csv`
- `../results/charts/local_size_sweep_gpu_ms_vs_local_size.png`
- `../results/charts/local_size_sweep_throughput_vs_local_size.png`
- `../results/charts/local_size_sweep_speedup_vs_ls64.png`
- `../results/charts/local_size_sweep_operation_ratio_summary.png`
- `../results/charts/local_size_sweep_status_overview.png`
- `../results/charts/local_size_sweep_test_setup.png`
- `../results/charts/benchmark_summary.png`
