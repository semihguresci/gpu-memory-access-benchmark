# Experiment 15: Bandwidth Saturation Sweep

## 1. Focus
- Find the workload size where simple contiguous kernels reach a sustained bandwidth plateau.

## 2. Question
- How large must a memory workload be before fixed dispatch overhead is mostly amortized?

## 3. Variants
- `read_only`
- `write_only`
- `read_write_copy`
- dense size sweep from small to large buffers

## 4. Method
- Reuse the simple contiguous memory modes from the baseline experiment with a denser size sweep.
- Keep timing limited to GPU dispatch so the plateau reflects kernel execution rather than staging cost.

## 5. Outputs
- GB/s vs size for each memory mode.
- Plateau onset estimate.
- Sustained-region median GB/s.

## 6. Interpretation
- The plateau is practical measured bandwidth for the tested device, not proof of theoretical peak.
- Later experiments should prefer sizes in this stable region when the goal is to study steady-state behavior.
