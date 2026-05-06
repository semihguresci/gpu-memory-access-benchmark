# Experiment 37 Results: Frame-to-Frame Coherence

## Run Status
- Benchmark status: latest export completed (`90/90` row correctness pass)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations, `8` frames per timed sample
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:16:55Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_171655Z.json)
- Sweep coverage: `3` variants x `6` problem sizes

## Key Measurements
- All `18` benchmark cases are represented in the refreshed export.
- At the largest tested `element_count=67108864`, the fastest median GPU time came from `variant=block_scramble` at `196.660224 ms`. Median GB/s: `21.840`. Median throughput: `2729941525.949`.
- At the same size, `variant=coherent_shift` remained close at `203.937056 ms`, while `variant=frame_random` dropped to `3351.646144 ms`, `1.281 GB/s`, and `160181262.858` throughput.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [frame-to-frame coherence relative](./results/tables/frame_to_frame_coherence_relative.csv)
- [frame-to-frame coherence stability](./results/tables/frame_to_frame_coherence_stability.csv)
- [frame-to-frame coherence summary](./results/tables/frame_to_frame_coherence_summary.csv)
- [frame-to-frame coherence median gpu ms](./results/charts/frame_to_frame_coherence_ms.png)
- [frame-to-frame coherence relative](./results/charts/frame_to_frame_coherence_relative.png)
- [frame-to-frame coherence stability](./results/charts/frame_to_frame_coherence_stability.png)
- [frame-to-frame coherence throughput](./results/charts/frame_to_frame_coherence_throughput.png)

## Interpretation
- In the current `8`-frame sequence, `coherent_shift` and `block_scramble` stayed close, with `block_scramble` ranging from `97.4%` to `103.7%` of coherent throughput across the sweep. `frame_random` was the clear outlier, reaching only `6.1%-31.0%` of coherent throughput and up to `16.43x` slower median GPU time.

## Limitations
- Results come from one GPU and driver stack.
- Each timed sample batches `8` consecutive frames, so these numbers reflect short sequence behavior rather than a single-frame dispatch in isolation.
- Reported GB/s and throughput follow this experiment's metric definitions; compare them within the experiment before comparing them across experiments.
