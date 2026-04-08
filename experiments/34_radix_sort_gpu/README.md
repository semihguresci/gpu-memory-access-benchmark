# Experiment 34: Radix Sort on GPU

## Folder Role
This folder stores experiment-local artifacts, run archives, and helper scripts for this benchmark.

## Canonical Documents
- [Experiment plan](../../docs/advanced_plans/01_radix_sort_gpu.md)
- [Results report](results.md)

## Local Contents
- `results.md`: current measured conclusions and artifact links
- `results/`: generated tables and charts
- `runs/`: archived raw benchmark exports
- `scripts/`: experiment-local collection and analysis helpers

## Overview
Implements a multi-pass GPU radix sort using three kernel stages per pass (count histogram,
prefix scan, scatter). Two digit-width variants are compared:

| Variant | Digit bits | Passes | Radix size |
|---------|-----------|--------|------------|
| `8bit_4pass` | 8 | 4 | 256 |
| `4bit_8pass` | 4 | 8 | 16 |

The scan pass runs as a single workgroup of 256 threads performing a sequential per-block
prefix followed by a Blelloch inclusive scan to compute global digit start offsets.
Problem sizes are capped at 65K elements (256 blocks) to keep the single-workgroup scan
practical without a hierarchical approach.

Repo-wide architecture, development planning, and implementation sequencing live under `../../docs/`.
