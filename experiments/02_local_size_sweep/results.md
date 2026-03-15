# Experiment 02 Results: Local Size Sweep

Run date: 2026-03-14  
Experiment spec: [experiment_plan.md](./experiment_plan.md)

## 1. Goal and Objective Check
This experiment targets a correctness-first local-size sweep to identify practical `local_size_x` recommendations for later experiments.

Status:
![Run Status Overview](./results/charts/local_size_sweep_status_overview.png)

Source table:
- [local_size_sweep_status_overview.csv](./results/tables/local_size_sweep_status_overview.csv)

Observed status:
- `600/600` rows passed correctness checks in the latest run.
- `gpu_ms` and `end_to_end_ms` coverage are both `100%`.

## 2. Test Setup
![Test Setup Overview](./results/charts/local_size_sweep_test_setup.png)

Source table:
- [local_size_sweep_test_setup.csv](./results/tables/local_size_sweep_test_setup.csv)
- [local_size_sweep_runs_index.csv](./results/tables/local_size_sweep_runs_index.csv)

Latest run setup (used by primary summary/charts):
- GPU: `NVIDIA GeForce RTX 2080 SUPER`
- Vulkan API: `1.4.325`
- Validation: disabled
- Warmup/timed: `2/5`
- Problem sizes: `2^14 .. 2^23` (`16384 .. 8388608`)
- Local sizes: `{32, 64, 128, 256, 512, 1024}`
- Dispatch count: `{1}`

## 3. Core Graphs
Primary local-size latency view:
![GPU Time vs Local Size](./results/charts/local_size_sweep_gpu_ms_vs_local_size.png)

Primary local-size throughput view:
![Throughput vs Local Size](./results/charts/local_size_sweep_throughput_vs_local_size.png)

Speedup versus baseline local size (`64`):
![Speedup vs LS64](./results/charts/local_size_sweep_speedup_vs_ls64.png)

Reference chart from the quick plot script:
![Benchmark Summary](./results/charts/benchmark_summary.png)

## 4. Key Tables
Primary summary table:
- [local_size_sweep_summary.csv](./results/tables/local_size_sweep_summary.csv)

Best local size per `(variant, problem_size)`:
- [local_size_sweep_best_local_size.csv](./results/tables/local_size_sweep_best_local_size.csv)

Speedup baseline table (`vs local_size_x=64`):
- [local_size_sweep_speedup_vs_ls64.csv](./results/tables/local_size_sweep_speedup_vs_ls64.csv)

Overall ranking table (geometric mean speedup):
- [local_size_sweep_local_size_ranking.csv](./results/tables/local_size_sweep_local_size_ranking.csv)

Operation ratio table (`contiguous_write / noop`):
- [local_size_sweep_operation_ratio.csv](./results/tables/local_size_sweep_operation_ratio.csv)

Pivot tables:
- [local_size_sweep_gpu_ms_pivot.csv](./results/tables/local_size_sweep_gpu_ms_pivot.csv)
- [local_size_sweep_throughput_pivot.csv](./results/tables/local_size_sweep_throughput_pivot.csv)

## 5. Interpretation vs Hypothesis
Hypothesis: very small groups underutilize hardware, very large groups increase pressure, and middle values should perform best.

Observed in latest run:
- Small sizes (`2^14 .. 2^16`) are often best at `local_size_x=32`.
- Medium/large sizes increasingly favor `256..1024`.
- For the largest sizes in this run, `1024` is consistently best for `contiguous_write`.
- Ranking result (contiguous_write): `local_size_x=512` is best overall by geometric-mean speedup vs `64`:
  - geometric mean speedup: `1.3146x`
  - median speedup: `1.4197x`

Peak throughput points:
- `contiguous_write`: `9.55e10 elem/s` at `problem_size=8388608`, `local_size_x=1024`
- `noop`: `2.08e11 elem/s` at `problem_size=8388608`, `local_size_x=512`

This partially supports the hypothesis: middle-to-large local sizes dominate the steady-state region, while very small local sizes can still win in overhead-sensitive small-size points.

## 6. Operation-Level Comparison (contiguous_write vs noop)
Ratio summary chart:
![Operation Ratio Summary](./results/charts/local_size_sweep_operation_ratio_summary.png)

Source table:
- [local_size_sweep_operation_ratio_summary.csv](./results/tables/local_size_sweep_operation_ratio_summary.csv)

From [local_size_sweep_operation_ratio.csv](./results/tables/local_size_sweep_operation_ratio.csv), median `gpu_ms` ratio (`contiguous_write / noop`) rises with local size:
- `32`: `1.008`
- `64`: `1.075`
- `128`: `1.224`
- `256`: `1.262`
- `512`: `1.311`
- `1024`: `1.369`

Interpretation:
- No-op cost drops faster than write cost as local size grows in this workload.
- Larger local sizes still improve absolute write throughput, but no-op-relative overhead increases.

## 7. Practical Rule Extracted
For this workload and GPU:
- Use `local_size_x=512` as the primary default for throughput-oriented follow-up experiments.
- Keep `1024` as a high-throughput alternative for larger problem sizes.
- For very small problem sizes, re-check `32`/`64` because overhead sensitivity can change the winner.

## 8. Limitations
- Current dataset covers one GPU model; ranking should be re-validated with additional devices.
- Dispatch count was fixed at `1` in current runtime configuration.
- Validation layers were disabled during measured runs.
