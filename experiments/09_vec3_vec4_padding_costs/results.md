# Experiment 09 Results: vec3 vs vec4 Padding Costs

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`75/75` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 128M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:35:27Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_143527Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_143527Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `5` problem sizes

## Key Measurements
- At the largest tested problem size (`problem_size=2097152`), `split_scalars` measured `10.263968 ms` and `17.980 GB/s`.
- `vec4` measured `36.060672 ms` and `5.583 GB/s`, while `vec3_padded` measured `63.173344 ms` and `4.249 GB/s`.
- Relative to `vec4`, `split_scalars` was `3.513x` faster and `vec3_padded` was `1.752x` slower in GPU time.
- Alignment waste at the largest size was `0%` for `split_scalars`, `9.09%` for `vec4`, and `45.45%` for `vec3_padded`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/vec3_vec4_padding_summary.csv)
- [Median GPU-time chart](./results/charts/vec3_vec4_padding_median_gpu_ms.svg)

## Interpretation
- The full-refresh run still matches the experiment hypothesis: more padding correlates with worse measured performance on this kernel.
- `split_scalars` is the clear winner in the current data, `vec4` is the middle ground, and `vec3_padded` remains the worst choice.

## Limitations
- Results come from one GPU and driver stack.
- The current artifact bundle is intentionally small for this experiment: one summary table and one chart.
