# Experiment 12 Results: Gather Access Pattern

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`20/20` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 32M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:41:40Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_144140Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_144140Z_full_refresh_20260329.json)
- Sweep coverage: `4` variants x `1` logical size

## Key Measurements
- `identity` set the baseline at `5.174624 ms` and `19.453 GB/s` for `logical_elements=8388608`.
- `clustered_random_256` stayed effectively tied at `5.220672 ms` and `19.282 GB/s`, only `1.009x` slower than `identity`.
- `block_coherent_32` measured `6.244704 ms` and `16.120 GB/s`, or `1.207x` slower than the baseline in this full-refresh run.
- `random_permutation` collapsed to `33.725536 ms` and `2.985 GB/s`, or `6.517x` slower than `identity`.
- `block_coherent_32` was the steadiest variant (`p95/median=1.024x`), while `identity` and `clustered_random_256` both showed larger tails at about `1.23x`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/gather_access_pattern_summary.csv)
- [Relative table](./results/tables/gather_access_pattern_relative.csv)
- [Stability table](./results/tables/gather_access_pattern_stability.csv)
- [Slowdown chart](./results/charts/gather_access_pattern_slowdown_vs_identity.png)
- [Stability chart](./results/charts/gather_access_pattern_stability_ratio.png)

## Interpretation
- Preserving broad locality is enough to stay near the identity baseline, but a full random permutation still destroys source-read performance on this GPU.
- This full-refresh snapshot still does not show a win for `block_coherent_32`; it trails both `identity` and `clustered_random_256`, even though it is the steadiest of the near-baseline variants.

## Limitations
- Results come from one GPU and driver stack.
- The run uses one logical problem size and no hardware counters, so locality conclusions are timing-based only.
