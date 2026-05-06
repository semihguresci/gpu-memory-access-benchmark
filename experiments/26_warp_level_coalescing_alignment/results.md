# Experiment 26 Results: Warp-Level Coalescing Alignment

## Run Status
- Benchmark status: latest `20260428_202621Z_2` collection completed (`30/30` row correctness pass across `6` cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T20:26:21Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_202621Z_2.json)
- Sweep coverage: `6` alignment offsets (`0`, `4`, `8`, `16`, `32`, `64` bytes) at `268435200` logical elements

## Key Measurements
- All `6` benchmark cases report `correctness_pass_rate=1.000000`.
- The fastest median GPU time was `alignment_offset_bytes=4` at `97.018656 ms`. Median GB/s: `22.134728`. Median throughput: `2766841049.622457`.
- The aligned `0`-byte baseline measured `106.774304 ms` and `20.112345 GB/s`, and the full sweep ranged from `97.018656 ms` (`4` bytes) to `107.657408 ms` (`16` bytes).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [warp level coalescing alignment summary](./results/tables/warp_level_coalescing_alignment_summary.csv)
- [warp level coalescing alignment relative](./results/tables/warp_level_coalescing_alignment_relative.csv)
- [warp level coalescing alignment stability](./results/tables/warp_level_coalescing_alignment_stability.csv)
- [warp level coalescing alignment footprint](./results/tables/warp_level_coalescing_alignment_footprint.csv)
- [warp level coalescing alignment median gbps](./results/charts/warp_level_coalescing_alignment_median_gbps.png)
- [warp level coalescing alignment median gpu ms](./results/charts/warp_level_coalescing_alignment_median_gpu_ms.png)
- [warp level coalescing alignment slowdown vs aligned](./results/charts/warp_level_coalescing_alignment_slowdown_vs_aligned.png)
- [warp level coalescing alignment stability ratio](./results/charts/warp_level_coalescing_alignment_stability_ratio.png)

## Interpretation
- This refreshed sweep does not show a monotonic penalty as the offset increases. `4`-byte and `8`-byte offsets outperformed the aligned baseline, while `16` bytes was only `0.827075%` slower.
- On this GPU and access pattern, the current offset effect is small enough that stronger transaction-boundary claims would need profiler support.

## Limitations
- Results come from one GPU/driver stack, one access pattern (`contiguous_shifted`), and one logical size (`268435200`).
- The kernel is a fixed `uint32_t` read-plus-write transform; different kernels may respond differently to the same offsets.
- No profiler capture is linked here, so the report should not attribute the current ranking to a specific hardware mechanism.
