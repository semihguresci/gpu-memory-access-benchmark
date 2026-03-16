# Experiment 09 Results

## Run and Test Snapshot
- Benchmark status: latest run completed (`36/36` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Debug --output-on-failure` passed (`17/17`)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 3 --warmup 1 --size 64M`
- Raw export timestamp (UTC): `2026-03-16T21:14:01Z`
- Latest collected run: `runs/nvidia_geforce_rtx_2080_super/20260316_211401Z_doc_run.json`

## Key Measurements
Largest tested size (`problem_size=1048576`, medians from current run):
- `split_scalars`: `4.634976 ms`, `19.9083 GB/s`
- `vec4`: `18.281280 ms`, `5.5064 GB/s`
- `vec3_padded`: `34.625440 ms`, `3.8763 GB/s`

Relative to `vec4` at `problem_size=1048576`:
- `vec3_padded`: `+89.40%` slower in `gpu_ms`
- `split_scalars`: `-74.65%` lower `gpu_ms` (faster)

## Layout Footprint Context
- `vec3_padded`: `64` storage bytes/particle, `44` logical bytes/particle, alignment waste `45.45%`
- `vec4`: `48` storage bytes/particle, `44` logical bytes/particle, alignment waste `9.09%`
- `split_scalars`: `44` storage bytes/particle, `44` logical bytes/particle, alignment waste `0%`

## Graphics
![Median GPU ms by Problem Size](./results/charts/vec3_vec4_padding_median_gpu_ms.svg)

## Data Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/vec3_vec4_padding_summary.csv)
- [Latest collected run](./runs/nvidia_geforce_rtx_2080_super/20260316_211401Z_doc_run.json)

## Interpretation and Limits
- This dataset is consistent with the experiment hypothesis: higher padding overhead (`vec3_padded`) corresponds to worse observed performance than tighter layouts.
- In this workload on this GPU, `split_scalars` is fastest across the tested sizes.
- Limits:
  - results come from one GPU/driver stack;
  - measurements use a Debug binary and `timed_iterations=3`, so variance characterization is limited;
  - conclusions should be revalidated on Release builds and additional hardware.
