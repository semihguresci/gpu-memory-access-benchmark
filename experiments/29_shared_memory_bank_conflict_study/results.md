# Experiment 29 Results: Shared Memory Bank Conflict Study

## Run Status
- Benchmark status: latest `20260428_185709Z` collection completed (`35/35` row correctness pass across `7` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T18:57:09Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_185709Z_2.json)
- Sweep coverage: `7` variants at the current configured problem size (`134217728` elements)

## Key Measurements
- All `7` benchmark cases are represented in the refreshed export.
- The fastest median GPU time came from `variant=stride_4` at `49.413696 ms`. Median GB/s: `21.729640`. Median throughput: `2716204997.092`.
- The worst conflict case in the current sweep was `stride_32` at `68.348384 ms`, which is a `1.344555x` slowdown versus the `stride_1` baseline and drops median GB/s to `15.709835`.
- The padded fix recovered most of that loss: `variant=padded_fix` (`shared_stride_elements=33`) measured `50.605056 ms`, only `0.995507x` of the `stride_1` GPU time and `1.004514x` of its GB/s.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.063766` for `stride_1`; `stride_32` reached `1.048142`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [bank conflict summary](./results/tables/shared_memory_bank_conflict_study_summary.csv)
- [bank conflict relative](./results/tables/shared_memory_bank_conflict_study_relative.csv)
- [bank conflict stability](./results/tables/shared_memory_bank_conflict_study_stability.csv)
- [bank conflict gpu ms](./results/charts/bank_conflict_gpu_ms.png)
- [bank conflict slowdown](./results/charts/bank_conflict_slowdown.png)
- [bank conflict stability](./results/charts/bank_conflict_stability.png)

## Interpretation
- On this GPU, the expected high-conflict `stride_32` case does degrade performance materially relative to the low-stride baselines.
- The padded `33`-element layout removes most of that penalty in the current configuration, which is the main engineering point of this experiment.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers one configured problem size only, so it does not show whether the conflict penalty changes with a smaller or larger shared-memory workload.
- The fastest case in this run was `stride_4`, not `stride_1`, so the conflict story here is clearest from the relative `stride_32` collapse and the `padded_fix` recovery rather than from a monotonic stride curve.
