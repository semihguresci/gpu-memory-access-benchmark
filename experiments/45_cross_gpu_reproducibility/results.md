# Experiment 45 Results: Cross-GPU Reproducibility

## Run Status
- Benchmark status: latest `20260428_205300Z_2` collection completed (`105/105` timed rows correctness pass across `21` benchmark cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T20:53:00Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_205300Z_2.json)
- Sweep coverage: `3` variants x `7` problem sizes (`1048576` to `134217728` elements)

## Key Measurements
- All `21` benchmark cases are represented in the refreshed export, every summary row reports `correctness_pass_rate=1.0`, and the checksum table shows `21/21` stable rows with `checksum_unique=1`.
- At the signature size `134217728`, `coalesced_reference` remained fastest at `53.332576 ms`. `stride8_probe` measured `349.553344 ms` (`6.554218x` slower) and `hashed_probe` measured `1139.304480 ms` (`21.362262x` slower).
- At the smallest tested `problem_size=1048576`, both stressed-access variants were already slower than `coalesced_reference`: `hashed_probe` at `4.197064x` and `stride8_probe` at `4.294137x`.
- The widest p95-to-median timing spread in the current sweep was `1.444735` for `hashed_probe` at `problem_size=1048576`.

## Artifact Links
- [Latest collected run JSON](./runs/nvidia_geforce_rtx_2080_super/20260428_205300Z_2.json)
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [cross gpu reproducibility checksums](./results/tables/cross_gpu_reproducibility_checksums.csv)
- [cross gpu reproducibility relative](./results/tables/cross_gpu_reproducibility_relative.csv)
- [cross gpu reproducibility signature](./results/tables/cross_gpu_reproducibility_signature.csv)
- [cross gpu reproducibility stability](./results/tables/cross_gpu_reproducibility_stability.csv)
- [cross gpu reproducibility summary](./results/tables/cross_gpu_reproducibility_summary.csv)
- [cross gpu reproducibility median gpu ms](./results/charts/cross_gpu_reproducibility_ms.png)
- [cross gpu reproducibility relative comparison](./results/charts/cross_gpu_reproducibility_relative.png)
- [cross gpu reproducibility stability ratio](./results/charts/cross_gpu_reproducibility_stability.png)
- [cross gpu reproducibility throughput](./results/charts/cross_gpu_reproducibility_throughput.png)

## Interpretation
- On this GPU, the refreshed baseline keeps a clear ordering: `coalesced_reference` first, `stride8_probe` second, `hashed_probe` last across the full sweep.
- The checksum and signature artifacts are useful as a reproducible baseline for later device-to-device comparisons, but this single-GPU refresh does not by itself establish cross-GPU reproducibility.

## Limitations
- Results come from one GPU and driver stack.
- This report documents a current baseline only; matching collections on additional GPUs are still required before drawing cross-device conclusions.
- Reported throughput and GB/s values follow this experiment's metric definitions and are most reliable for within-experiment comparisons.
