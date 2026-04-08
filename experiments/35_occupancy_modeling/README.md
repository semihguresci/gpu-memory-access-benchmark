# Experiment 35: Occupancy Modeling

## Folder Role
This folder stores experiment-local artifacts, run archives, and helper scripts for this benchmark.

## Canonical Documents
- [Experiment plan](../../docs/advanced_plans/08_occupancy_modeling.md)
- [Results report](results.md)

## Local Contents
- `results.md`: current measured conclusions and artifact links
- `results/`: generated tables and charts
- `runs/`: archived raw benchmark exports
- `scripts/`: experiment-local collection and analysis helpers

## Overview
Measures how shared-memory pressure affects GPU compute throughput by varying the amount
of shared memory allocated per workgroup while keeping the computational work constant.

| Variant | Shared memory per WG | Expected occupancy effect |
|---------|--------------------|--------------------------|
| `low_smem` | 1 KB (256 uints) | Minimal — many WGs can co-reside |
| `medium_smem` | 8 KB (2048 uints) | Moderate — fewer WGs per CU |
| `high_smem` | 32 KB (8192 uints) | Maximum — typically 1 WG per CU |

Each variant performs a simple round-trip transformation through shared memory (read input →
process in shared scratch → write output) to prevent dead-code elimination while keeping
arithmetic cost negligible relative to memory access.

Repo-wide architecture, development planning, and implementation sequencing live under `../../docs/`.
