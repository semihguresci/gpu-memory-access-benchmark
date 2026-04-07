# Advanced Investigations Roadmap (Lecture Notes Track)

## 1. Why This Track Exists
The first 25 experiments build strong foundations.
This follow-up track is designed to convert those foundations into GPU systems and rendering-oriented portfolio evidence.

Bridge objective:
- from isolated benchmark literacy
- to composable GPU subsystem design
- to architecture-aware interpretation suitable for engine and DevTech work

## 2. How to Use This Roadmap
Recommended process:
1. use the completed core 25 experiments, plus the extension studies in Experiments 26-33, as the measurement baseline
2. identify strongest measured themes (3 to 5)
3. choose a focused advanced subset
4. execute deeply and compare variants rigorously

Suggested project labeling:
- Project 0.3: focused subset of advanced topics
- Project 0.4: broader multi-topic extension
- Project 1.0: full advanced track with cross-GPU comparisons

## 3. Advanced Themes
Primary themes covered:
1. GPU sorting and data reordering
2. spatial acceleration structure layout
3. visibility and culling pipelines
4. persistent work scheduling
5. subgroup and wave-level operations
6. async compute overlap
7. occupancy-guided modeling
8. rendering-oriented memory system studies

## 4. Investigation Set (01-12)
Detailed lecture-note plans for each investigation live in:
- `docs/10_advanced_investigation_plans_index.md`

Investigation list:
1. Radix sort on GPU
2. BVH node layout experiments
3. Frustum culling vs clustered culling
4. Tiled light assignment
5. Persistent threads and work queues
6. Subgroup operations study
7. Async compute overlap
8. Occupancy modeling against vendor guidance
9. Memory system study for ray-friendly layouts
10. Frame-to-frame coherence studies
11. GPU-driven pipeline building blocks
12. Reproducibility and cross-GPU comparison

## 5. Recommended Execution Phases
### Phase A: Core Primitive Depth
- radix sort
- subgroup operations
- occupancy modeling

### Phase B: Rendering-Relevant Data Systems
- tiled light assignment
- frustum vs clustered culling
- GPU-driven building blocks

### Phase C: Architecture-Depth Extensions
- BVH layout
- ray-friendly layouts
- frame-to-frame coherence

### Phase D: Systems and Platform Depth
- persistent work queues
- async overlap
- cross-GPU comparison

## 6. Deliverable Contract for Each Advanced Study
Minimum outputs:
- one core chart
- one summary table
- one concise conclusions page with limitations

Recommended folder shape:
```text
advanced/NN_name/
|- README.md
|- theory.md
|- experiment_plan.md
|- shader/
|- src/
|- configs/
|- results/
|  |- raw/
|  |- processed/
|  |- charts/
|  `- notes/
`- conclusions.md
```

## 7. Interpretation Guardrails
- Separate measured facts from interpretation.
- Label inferred occupancy/resource effects as inference, not direct measurement.
- Avoid universal claims from one hardware sample.
- Include failure modes and non-winning variants in conclusions.

## 8. Final Positioning
Core track message:
- "I understand GPU performance foundations."

Advanced track message:
- "I can apply those foundations to design and analyze practical GPU systems that matter in real rendering and compute pipelines."
