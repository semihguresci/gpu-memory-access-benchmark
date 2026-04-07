# Experiment 28: Device-Local vs Host-Visible Heap Placement

## 1. Focus
- Compare direct host-visible storage buffers against staged device-local buffers.

## 2. Question
- When does device-local placement improve kernel throughput enough to justify staging overhead?

## 3. Variants
- `host_visible_direct`
- `device_local_staged`

## 4. Method
- Run the same copy-style kernel for both placements.
- Record `upload_ms`, `dispatch_ms`, `readback_ms`, and `end_to_end_ms` for the same logical payload.

## 5. Outputs
- Median GPU dispatch time by placement.
- End-to-end time by placement.
- Effective dispatch GB/s by placement.

## 6. Interpretation
- Faster dispatch on device-local buffers does not automatically mean faster end-to-end runtime.
- This experiment is most important on discrete desktop GPUs.
