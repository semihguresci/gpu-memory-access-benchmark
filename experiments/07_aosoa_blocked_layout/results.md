# Experiment 07 Results: AoSoA Blocked Layout

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`120/120` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 512M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:33:34Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_143334Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_143334Z_full_refresh_20260329.json)
- Sweep coverage: `6` variants x `4` problem sizes

## Key Measurements
- At the largest tested problem size (`problem_size=8000000`), `soa` measured `32.271904 ms` and `19.831 GB/s`.
- `aosoa_b32` stayed close at `33.624544 ms` and `19.034 GB/s`, only `1.042x` slower than `soa`.
- Smaller blocks degraded steadily: `aosoa_b16` was `1.122x` slower than `soa`, `aosoa_b8` was `1.564x` slower, and `aosoa_b4` was `2.339x` slower.
- `aos` remained the clear worst case at `448.698080 ms` and `2.282 GB/s`, or `13.904x` slower than `soa`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/aosoa_blocked_layout_summary.csv)
- [GPU time chart](./results/charts/aosoa_blocked_layout_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/aosoa_blocked_layout_gbps_vs_size.png)

## Interpretation
- On this GPU, AoSoA only pays when blocks stay large enough to preserve the memory behavior that makes SoA effective. `aosoa_b32` is still effectively tied with SoA in the full-refresh run.
- Once the block size shrinks, the layout penalty returns quickly. That trend remains monotonic from `b32` down to `b4` in the current dataset.

## Limitations
- Results come from one GPU and driver stack.
- The experiment measures a single access pattern; different kernels could shift the best block size.
