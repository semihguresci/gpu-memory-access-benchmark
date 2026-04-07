# Experiment 20 Results: Barrier Synchronization Cost

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`50/50` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:04:08Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140408Z_full_refresh_20260405.json)
- Sweep coverage: `10` variants x current configured problem sizes

## Key Measurements
- All `10` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=4194304`, the fastest median GPU time came from `placement=flat_loop, local_size_x=256` at `1.497792 ms`. Median GB/s: `22.403`. Median throughput: `2800324744.691`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [barrier synchronization cost relative](./results/tables/barrier_synchronization_cost_relative.csv)
- [barrier synchronization cost stability](./results/tables/barrier_synchronization_cost_stability.csv)
- [barrier synchronization cost summary](./results/tables/barrier_synchronization_cost_summary.csv)
- [barrier synchronization cost estimated gbps](./results/charts/barrier_synchronization_cost_estimated_gbps.png)
- [barrier synchronization cost median gpu ms](./results/charts/barrier_synchronization_cost_median_gpu_ms.png)
- [barrier synchronization cost speedup vs flat](./results/charts/barrier_synchronization_cost_speedup_vs_flat.png)
- [barrier synchronization cost stability ratio](./results/charts/barrier_synchronization_cost_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `35.96%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
