# Experiment 09 Results: Vec3 Vec4 Padding Costs

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`45/45` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:53:59Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135359Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=524288`, the fastest median GPU time came from `variant=split_scalars` at `2.487456 ms`. Median GB/s: `18.548`. Median throughput: `210772773.468`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [vec3 vec4 padding summary](./results/tables/vec3_vec4_padding_summary.csv)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `566.43%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
