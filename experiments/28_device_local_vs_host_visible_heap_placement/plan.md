# Experiment 28: Device-Local vs Host-Visible Heap Placement

## 1. Focus
- Compare direct host-visible storage buffers against staged device-local buffers.

## 2. Question
- When does device-local placement improve kernel throughput enough to justify staging overhead?

## 3. Variants
- `host_visible_direct`
- `device_local_staged`
- Reuse-amortization groups `*_reuse_k{2,4,16,64}` for both plans: upload once, run K identical kernels with inter-dispatch write barriers, read back once. The base variants are the K = 1 case and keep their names.

## 4. Method
- Run the same copy-style kernel for both placements.
- Record `upload_ms`, `chain_ms` (the K-kernel group duration including inter-dispatch barriers), `readback_ms`, and `end_to_end_ms` for the same logical payload. Base rows keep the historical `dispatch_ms` key equal to the single-kernel chain duration.
- Repeated identical kernels write the same output region; every reuse row therefore records `inter_dispatch_dependencies = K - 1` and a write-after-write barrier between consecutive dispatches inside the timed chain. Identical rewrites cannot prove each repetition executed; the ordered barrier chain establishes the dependency, and row notes document this scope.
- All K kernels in a group share one CPU fill and one readback per timed iteration, then per-iteration validation; the analysis derives per-kernel cost from the `K = 1..64` medians rather than one-to-one notes.
- Budget all six simultaneously allocated buffers. Per-buffer payload is at most one sixth of the CLI scratch budget; notes distinguish all six live allocations from the two buffers active in direct access and four active in staging.
- Export each buffer's actual memory type, heap, and property flags, total allocated bytes including Vulkan allocation padding, and active/live payload bytes. The scratch budget covers buffer payload, excluding allocation padding; physical residency is not measured.
- The primary `gpu_ms` is the sum of separate upload, dispatch-chain, and readback timestamp intervals. CPU submission gaps are outside this sum and inside the measured end-to-end wall time.

## 5. Outputs
- Median summed GPU path time by placement and reuse group, with upload/chain/readback stage times in row notes.
- End-to-end time by placement and reuse group.
- Useful kernel payload GB/s over the summed GPU path time. This counts K input and output word pairs per element-kernel rather than physical bus traffic.
- Reuse-amortization table: GPU path per kernel for staged (amortized) versus direct at each K, plus the measured staged/direct ratio locating the crossover.

## 6. Interpretation
- Faster dispatch on device-local buffers does not automatically mean faster end-to-end runtime.
- This experiment is most important on discrete desktop GPUs.
- Requested `HOST_VISIBLE` and `DEVICE_LOCAL` properties can refer to the same heap or memory type; exported actual placement must support any interpretation about host/device transfers.
