# Experiment 41 Results: Subgroup Operations Study

## Run Status
- Benchmark status: latest `20260429_181046Z` collection completed; `70/70` recorded rows passed correctness.
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `timed_iterations=5`, `warmup_iterations=2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-29T18:10:46Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260429_181046Z_2.json)
- Sweep coverage: `2` variants x `7` problem sizes (`14` benchmark cases)

## Key Measurements
- All `14` benchmark cases are represented in the current export.
- At the largest tested `problem_size=134217728`, the fastest median GPU time came from `variant=shared_baseline` at `66.093056 ms`. The subgroup-intrinsics path measured `66.814048 ms`, or `0.989209x` of the baseline throughput.
- The largest observed subgroup advantage occurs at `problem_size=33554432`, where `subgroup_intrinsics` measures `11.957952 ms` versus `12.381408 ms` for `shared_baseline` and reaches `1.035412x` the baseline throughput.
- The largest observed subgroup regression occurs at `problem_size=67108864`, where `subgroup_intrinsics` measures `27.368000 ms` versus `24.847904 ms` for `shared_baseline`, or `0.907918x` of the baseline throughput.
- The largest p95-to-median GPU-time ratio in the current stability table is `2.029843` for `shared_baseline` at `problem_size=8388608`; the subgroup path peaks at `1.204080` at `problem_size=33554432`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [subgroup operations study relative](./results/tables/subgroup_operations_study_relative.csv)
- [subgroup operations study stability](./results/tables/subgroup_operations_study_stability.csv)
- [subgroup operations study summary](./results/tables/subgroup_operations_study_summary.csv)
- [subgroup operations study median gpu ms](./results/charts/subgroup_operations_study_ms.png)
- [subgroup operations study relative](./results/charts/subgroup_operations_study_relative.png)
- [subgroup operations study stability](./results/charts/subgroup_operations_study_stability.png)
- [subgroup operations study throughput](./results/charts/subgroup_operations_study_throughput.png)

## Interpretation
- On this GPU, after changing the shared-memory path into a cooperative baseline, subgroup intrinsics behave like a near-parity alternative rather than a broad win.
- The corrected sweep shows small subgroup gains at several smaller and mid-size points, but the largest tested sizes include regressions, so the defensible takeaway is that memory-system effects dominate this benchmark more than the choice of reduction primitive.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s follows the experiment's recorded logical-payload byte accounting (`gbps_mode=logical_payload_bytes`), so compare it within this experiment before comparing it across experiments.
- The current sweep covers only the configured `1048576` to `134217728` element counts and the two shipped variants.
- The shared baseline is a cooperative shared-memory reduction that keeps useful work closer to the subgroup path, but it is still a software baseline rather than a hardware-equivalent implementation of subgroup collectives.
