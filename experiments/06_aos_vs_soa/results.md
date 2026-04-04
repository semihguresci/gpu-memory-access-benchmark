# Experiment 06 Results: AoS vs SoA

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`10/10` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 64M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:19:02Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_141902Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_141902Z_full_refresh_20260329.json)
- Sweep coverage: `2` variants x `1` problem size in the current raw export

## Key Measurements
- The current full-refresh snapshot covers `problem_size=1000000` and measured `soa` at `2.472128 ms` and `19.416 GB/s`.
- The corresponding `aos` run measured `70.268320 ms` and `0.911 GB/s`.
- `soa` was `28.424x` faster than `aos` in GPU time on this access pattern.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/aos_vs_soa_summary.csv)
- [GPU time chart](./results/charts/aos_vs_soa_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/aos_vs_soa_gbps_vs_size.png)

## Interpretation
- This access pattern is decisively field-wise on the RTX 2080 SUPER. SoA keeps memory transactions aligned with the kernel, while AoS collapses effective bandwidth.
- The result remains strong enough that later layout experiments should continue to treat SoA as the practical baseline unless a different access pattern is explicitly being optimized.

## Limitations
- Results come from one GPU and driver stack.
- The current headline export contains one problem size; older archived runs include larger sizes but are not the report baseline here.
