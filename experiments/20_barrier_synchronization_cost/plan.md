# Experiment 20: Barrier and Synchronization Cost

## 1. Focus
- Isolate the runtime cost of workgroup barriers and synchronization placement.

## 2. Question
- How much overhead comes from barriers themselves, and does that cost depend on how work is tiled?

## 3. Variants
- `flat_loop_b0/b1/b2/b4/b8`
- `tiled_regions_b0/b1/b2/b4/b8`

## 4. Method
- Publish one immutable shared seed per workgroup using one mandatory barrier in every variant. Keeping the seed immutable prevents a faster subgroup from overwriting data still in use by another subgroup.
- Execute the same eight arithmetic phases and shared-seed reads for every invocation, producing exactly the same output in all ten variants. Validate every timed iteration against one common CPU reference.
- Insert additional barriers every 1, 2, 4, or 8 phases, or none for `b0`. Including seed publication, the exact barrier counts per workgroup are 9, 5, 3, 2, and 1 respectively. Flat and tiled variants change loop placement only.
- The additional barriers are redundant for correctness and measure synchronization overhead around this fixed workload. Shared access counts describe source operations; a driver may hoist immutable seed reads.

## 5. Outputs
- Median GPU time by synchronization strategy.
- Incremental overhead relative to `b0`, which retains its required seed-publication barrier.
- Placement sensitivity between flat and tiled execution shapes.

## 6. Interpretation
- A barrier cost is only meaningful relative to the work it protects.
- Barrier timing depends on this workload and compiler scheduling; it does not establish a universal barrier latency or the cost of arbitrary producer/consumer communication.
