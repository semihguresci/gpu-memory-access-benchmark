# Experiment 36 Results: BVH Node Layout

## Run Status
- Benchmark status: latest export completed (`90/90` row correctness pass)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:13:37Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_171337Z.json)
- Sweep coverage: `3` variants x `6` problem sizes

## Key Measurements
- All `18` benchmark cases are represented in the refreshed export.
- At the largest tested `node_count=5592320`, the fastest median GPU time came from `variant=compact32_sequential` at `13.951616 ms`. Median GB/s: `14.430`. Median throughput: `400836720.277`.
- At the same size, `variant=padded64_sequential` measured `42.417504 ms` (`3.04x` slower than `compact32_sequential`) and `variant=compact32_hashed` measured `211.283584 ms` (`15.14x` slower).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [bvh node layout relative](./results/tables/bvh_node_layout_relative.csv)
- [bvh node layout stability](./results/tables/bvh_node_layout_stability.csv)
- [bvh node layout summary](./results/tables/bvh_node_layout_summary.csv)
- [bvh node layout gbps](./results/charts/bvh_node_layout_gbps.png)
- [bvh node layout median gpu ms](./results/charts/bvh_node_layout_ms.png)
- [bvh node layout relative](./results/charts/bvh_node_layout_relative.png)
- [bvh node layout stability](./results/charts/bvh_node_layout_stability.png)

## Interpretation
- In this fixed-work traversal kernel on this GPU, the compact sequential layout stayed ahead across the full sweep. The padded `64`-byte layout was consistently about `3.02x-3.27x` slower, while hashed access fell to `6.6%-39.9%` of the compact sequential GB/s range.

## Limitations
- Results come from one GPU and driver stack.
- This benchmark keeps per-node useful work constant, but it is still a synthetic traversal proxy rather than a full BVH traversal workload.
- Reported GB/s and throughput follow this experiment's metric definitions; compare them within the experiment before comparing them across experiments.
