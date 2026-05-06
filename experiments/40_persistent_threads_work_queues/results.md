# Experiment 40 Results: Persistent Threads and Work Queues

## Run Status
- Benchmark status: latest archived collection completed; `120/120` recorded rows passed correctness.
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `timed_iterations=5`, `warmup_iterations=2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:24:06Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_172406Z.json)
- Sweep coverage: `4` variants x `6` problem sizes (`24` benchmark cases)

## Key Measurements
- All `24` benchmark cases are represented in the current export.
- At the largest tested `problem_size=67108864`, the fastest median GPU time came from `variant=static_partitioned_uniform_cost` at `24.723776 ms`. Median GB/s: `21.715`. Median throughput: `2714345252.117`.
- Across the current relative table, the persistent-queue variants reached `21.739%` to `26.191%` of the matching static-partitioned throughput, with median GPU-time slowdowns between `3.818x` and `4.600x`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [persistent threads work queues relative](./results/tables/persistent_threads_work_queues_relative.csv)
- [persistent threads work queues stability](./results/tables/persistent_threads_work_queues_stability.csv)
- [persistent threads work queues summary](./results/tables/persistent_threads_work_queues_summary.csv)
- [persistent threads work queues median gpu ms](./results/charts/persistent_threads_work_queues_ms.png)
- [persistent threads work queues relative](./results/charts/persistent_threads_work_queues_relative.png)
- [persistent threads work queues stability](./results/charts/persistent_threads_work_queues_stability.png)
- [persistent threads work queues throughput](./results/charts/persistent_threads_work_queues_throughput.png)

## Interpretation
- On this refreshed sweep, scheduling mode is the dominant effect: `static_partitioned_*` stays ahead of both persistent-queue variants across every measured size and both distributions.
- The current data does not show the persistent queue closing that gap on this GPU; the distribution change is secondary relative to the scheduling-mode split in the measured range.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s follows the experiment's recorded estimated global-byte accounting (`gbps_mode=estimated_global_bytes`), so compare it within this experiment before comparing it across experiments.
- The current sweep covers only the configured `262144` to `67108864` task counts and the four shipped variants.
