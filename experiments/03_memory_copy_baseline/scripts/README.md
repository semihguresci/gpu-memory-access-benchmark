# Experiment 03 Scripts

These scripts are local to Experiment 03 and read/write files under:
- `../results/tables/`
- `../results/charts/`
- `../runs/`

## Usage
From repository root:

```powershell
python experiments/03_memory_copy_baseline/scripts/collect_run.py
python experiments/03_memory_copy_baseline/scripts/analyze_memory_copy_baseline.py
python experiments/03_memory_copy_baseline/scripts/plot_results.py
```

Root-level convenience scripts:

```powershell
python scripts/run_experiment_data_collection.py --experiment 03_memory_copy_baseline
python scripts/generate_experiment_artifacts.py --experiment 03_memory_copy_baseline
```
