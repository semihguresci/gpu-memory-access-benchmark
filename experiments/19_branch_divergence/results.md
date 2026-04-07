# Experiment 19 Results: Branch Divergence

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`30/30` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:03:15Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140315Z_full_refresh_20260405.json)
- Sweep coverage: `6` variants x current configured problem sizes

## Key Measurements
- All `6` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=8388608`, the fastest median GPU time came from `variant=random_p75, local_size_x=256` at `3.027392 ms`. Median GB/s: `22.167`. Median throughput: `2770902479.758`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [branch divergence relative](./results/tables/branch_divergence_relative.csv)
- [branch divergence stability](./results/tables/branch_divergence_stability.csv)
- [branch divergence summary](./results/tables/branch_divergence_summary.csv)
- [branch divergence median gbps](./results/charts/branch_divergence_median_gbps.png)
- [branch divergence median gpu ms](./results/charts/branch_divergence_median_gpu_ms.png)
- [branch divergence slowdown vs uniform true](./results/charts/branch_divergence_slowdown_vs_uniform_true.png)
- [branch divergence stability ratio](./results/charts/branch_divergence_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `34.35%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
