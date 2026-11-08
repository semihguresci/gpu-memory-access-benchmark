# Experiment 22: Prefix Sum Scan

## Folder Role
This folder stores experiment-local artifacts, run archives, and helper scripts for this benchmark.

## Canonical Documents
- [Experiment plan](plan.md)
- [Results report](results.md)

## Local Contents
- `results.md`: current measured conclusions and artifact links
- `results/`: generated tables and charts
- `runs/`: archived raw benchmark exports
- `scripts/`: experiment-local collection and analysis helpers

Repo-wide architecture, development planning, and implementation sequencing live under `../../docs/`.

## Metric Accounting

`throughput` counts completed logical scan results per second. `gbps` uses an
estimate of shader-source storage-buffer requests across all three timed
dispatches, including auxiliary buffers. For `N` logical elements and `B` blocks:

- Main scan: `8N + 4B` bytes (input read, output write, and block totals).
- Block-total scan: `8B + 4` bytes (totals read, prefixes write, and one scratch total).
- Carry application: `8N + 1024(B - 1)` bytes (output read/write and one carry read
  per invocation in every nonfirst 256-invocation workgroup, including inactive tail lanes).

The total is `16N + 12B + 4 + 1024(B - 1)` bytes. Row notes and derived summaries
record this model and the estimated byte count. This is not measured physical
DRAM bandwidth: compiler broadcast, caching, coalescing, and shared-memory
requests are outside the model. Use GPU time or logical throughput when comparing
time to solve the same scan, since batching changes the auxiliary request count.
