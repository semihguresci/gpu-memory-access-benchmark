# Experiment 03 Results: Memory Copy Baseline

Run date: 2026-03-15 (latest full run exported at `2026-03-15T08:47:14Z`)  
Experiment spec: [experiment_plan.md](./experiment_plan.md)

## 1. Goal and Status
This experiment measures raw memory throughput characteristics for:
- `read_only`
- `write_only`
- `read_write_copy`

Correctness status from latest full run:
- `165/165` rows passed correctness checks (`100%` pass rate)

Source tables:
- [memory_copy_baseline_status_overview.csv](./results/tables/memory_copy_baseline_status_overview.csv)
- [memory_copy_baseline_runs_index.csv](./results/tables/memory_copy_baseline_runs_index.csv)

## 2. Test Setup (Latest Full Run)
- GPU: `NVIDIA GeForce RTX 2080 SUPER`
- Vulkan API: `1.4.325`
- Validation: disabled
- Warmup/timed iterations: `2/5`
- Scratch size: `1G`
- Problem-size sweep: `1 MiB .. 1 GiB` (`262,144 .. 268,435,456` float elements)
- Variants: `read_only`, `write_only`, `read_write_copy`

## 3. Core Charts
Primary effective-bandwidth view:
![Memory Copy Baseline GB/s vs Size](./results/charts/memory_copy_baseline_gbps_vs_size.png)

Primary dispatch-time view:
![Memory Copy Baseline GPU ms vs Size](./results/charts/memory_copy_baseline_gpu_ms_vs_size.png)

Quick summary chart:
![Memory Copy Baseline Quick Summary](./results/charts/benchmark_summary.png)

## 4. Key Tables
- [memory_copy_baseline_summary.csv](./results/tables/memory_copy_baseline_summary.csv)
- [memory_copy_baseline_multi_run_summary.csv](./results/tables/memory_copy_baseline_multi_run_summary.csv)
- [memory_copy_baseline_runs_index.csv](./results/tables/memory_copy_baseline_runs_index.csv)

## 5. Key Observations (Latest Full Run)
At the largest tested size (`1 GiB`):
- `read_only`: `938.17 GB/s` (`gpu_ms_median = 1.1445`)
- `read_write_copy`: `438.47 GB/s` (`gpu_ms_median = 4.8977`)
- `write_only`: `391.04 GB/s` (`gpu_ms_median = 2.7459`)

General pattern:
- All modes scale up with size and approach a high-throughput region at larger working sets.
- `read_only` reports the highest effective GB/s in this setup.
- `read_write_copy` is lower than `read_only` and above `write_only` in this run, reflecting combined read+write traffic behavior.

## 6. Limitations
- Current full dataset is from one GPU/driver configuration.
- Validation layers were disabled for measured runs.
- Effective GB/s is computed from kernel-byte accounting; it is not a direct DRAM hardware counter.

## 7. Reproducibility
Collect a full run:
```powershell
python scripts/run_experiment_data_collection.py --experiment 03_memory_copy_baseline --size 1G --label full_1g
```

Regenerate artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 03_memory_copy_baseline --collect-run
```
