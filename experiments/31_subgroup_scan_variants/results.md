# Experiment 31 Results: Subgroup Scan Variants

## Run Status
- Benchmark status: latest `20260428_192000Z` collection completed (`30/30` row correctness pass across `6` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T19:20:00Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_192000Z_2.json)
- Sweep coverage: `2` variants x `3` `items_per_thread` settings at the current configured problem size (`134217728` elements)

## Key Measurements
- All `6` benchmark cases are represented in the refreshed export.
- The fastest median GPU time came from `shared_block_scan_items_1` at `51.356608 ms`. Median GB/s: `20.907569`. Median throughput: `2613446121.675`.
- `subgroup_block_scan_items_1` stayed close at `51.679360 ms`, which is `0.993755x` of the shared-block-scan speedup baseline for that setting.
- At `items_per_thread=4`, the subgroup path was slower: `3920.248672 ms` versus `3572.119616 ms`, or `0.911197x` of the shared baseline throughput.
- At `items_per_thread=8`, the subgroup path was also slower: `1175.649920 ms` versus `1164.463904 ms`, or `0.990485x` of the shared baseline throughput.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.126203` for `subgroup_block_scan_items_4`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [subgroup scan summary](./results/tables/subgroup_scan_variants_summary.csv)
- [subgroup scan relative](./results/tables/subgroup_scan_variants_relative.csv)
- [subgroup scan stability](./results/tables/subgroup_scan_variants_stability.csv)
- [subgroup scan gpu ms](./results/charts/subgroup_scan_gpu_ms.png)
- [subgroup scan speedup](./results/charts/subgroup_scan_speedup.png)
- [subgroup scan stability](./results/charts/subgroup_scan_stability.png)

## Interpretation
- On this GPU and with the current kernels, the subgroup-assisted scan path does not beat the shared-block baseline in the current sweep.
- The largest performance loss appears at `items_per_thread=4`, while `items_per_thread=1` is effectively near-parity between the two implementations.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers one configured problem size only, so it does not show whether the ranking changes across different total scan lengths.
- This report reflects the current `items_per_thread` variants only; it does not isolate whether the large slowdown at `4` and `8` comes from subgroup behavior itself or from the broader structure of those kernel variants.
