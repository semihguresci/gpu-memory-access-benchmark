# Experiment 43 Results: Ray-Friendly Memory Layouts

## Run Status
- Benchmark status: latest `20260428_203443Z_2` collection completed; all `20/20` benchmark cases in the refreshed sweep report `correctness_pass_rate=1.0`
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `4` variants x `5` problem sizes, `5` timed iterations, `2` warmup iterations
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T20:34:43Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_203443Z_2.json)

## Key Measurements
- `soa32_sequential` is the fastest variant at every tested size. Its strongest relative result is at `1048576` primitives, where median GPU time drops from `5.178624 ms` for `aos64_sequential` to `2.551520 ms` and median throughput rises from `202481585.842108` to `410961309.337179` (`2.029623x`)
- Under sequential access, `soa32_sequential` stays `36.69%` to `50.73%` faster than `aos64_sequential`, with median estimated bandwidth between `13.688207` and `14.806120 GB/s`
- Hashed access is the dominant penalty. At `5592320` primitives, `aos64_hashed` reaches `266.903456 ms` (`11.489072x` slower than `aos64_sequential`) and `soa32_hashed` reaches `203.771904 ms`; relative to their own sequential layouts, hashed access is about `11.49x` slower for `aos64` and `13.85x` slower for `soa32`

## Artifact Links
- [Latest runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_203443Z_2.json)
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [ray friendly memory layouts relative](./results/tables/ray_friendly_memory_layouts_relative.csv)
- [ray friendly memory layouts stability](./results/tables/ray_friendly_memory_layouts_stability.csv)
- [ray friendly memory layouts summary](./results/tables/ray_friendly_memory_layouts_summary.csv)
- [ray friendly memory layouts median gpu ms](./results/charts/ray_friendly_memory_layouts_ms.png)
- [ray friendly memory layouts relative](./results/charts/ray_friendly_memory_layouts_relative.png)
- [ray friendly memory layouts stability](./results/charts/ray_friendly_memory_layouts_stability.png)
- [ray friendly memory layouts throughput](./results/charts/ray_friendly_memory_layouts_throughput.png)

## Interpretation
- On this GPU, coherent traversal benefits substantially from the compact `soa32` layout, while the hashed mapping penalty is larger than the layout effect itself
- Layout still matters inside the hashed regime, but much less than access coherence; the current sweep supports "coherence first, layout second" more strongly than a universal SoA win

## Limitations
- Results come from one GPU and driver stack
- The hashed mapping is only one incoherent access pattern; other ray distributions or traversal kernels may change the ranking
- Reported GB/s values follow this experiment's own traffic model and are most useful for comparisons within this experiment
