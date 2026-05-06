# Experiment 17: Tile Size Sweep

## 1. Focus
- Tune the tile shape used by the shared-memory stencil path.

## 2. Question
- Which tile size gives the best tradeoff between reuse, occupancy, and local-memory overhead?

## 3. Variants
- `shared_tiled` with several tile sizes
- `direct_global` reference

## 4. Method
- Reuse the tiled stencil workload from Experiment 16 and sweep tile size only.
- Hold the stencil radius, output semantics, and timing path fixed across the sweep.

## 5. Outputs
- Median GPU time by tile size.
- Best tile-size recommendation.
- Speedup relative to the direct-global reference.

## 6. Interpretation
- Tile size is a tuning parameter, not a universal constant.
- The winning point should be treated as hardware- and workload-specific evidence.
