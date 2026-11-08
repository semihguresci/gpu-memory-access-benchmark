# Advanced Investigation 16: Buffers, Texel Buffers, Storage Images, and Sampled Images

## 1. Lecture Focus
- Concept: The same integer 2D dataset served through four resource representations.
- Why this matters: Buffers expose flat addressing the developer controls; texel buffers, storage images, and sampled images hand the implementation a tiled, format-aware layout that may be faster for 2D locality or slower under scatter. Engines choose representations without measurement support; this experiment supplies it.
- Central question: For identical coordinates, fetch counts, arithmetic, and outputs, how much does the storage backend change dispatch time under neighborhood-local and randomly-scattered access?

## 2. Learning Objectives
By the end of this investigation, you should be able to:
- name the descriptor and layout obligations of storage images and sampled images
- design a representation experiment that holds coordinates and fetch counts constant across backends
- stage and measure the one-time upload and layout-transition costs of images separately from dispatch time
- state the reproduction scope: one typed monochrome 32-bit format, one mipmap level, no filtering or format conversion

## 3. Theory Primer (Lecture Notes)
- Row-major and column-major buffers are the developer-visible flat layouts; storage images expose the optimized tiling the driver picked at upload time; sampled images run texelFetch through a sampler descriptor with the same typed data.
- The sampled image stays in SHADER_READ_ONLY_OPTIMAL, the storage image in GENERAL; those layouts are recorded in row notes.
- Upload and layout transitions happen once per (size, representation) and are measured as separate stages, exported in row notes, and excluded from the timed dispatch.
- The dataset content is a pure function of the logical 2D coordinate, so row-major storage of coordinates (x, y), the transposed placement, and both images hold identical values at identical coordinates; the CPU oracle recreates each placement exactly.

## 4. Hypothesis
Neighborhood-local fetches favor representations with 2D-aware tiling (images) once the working set exceeds cache; randomly scattered fetches flatten representation differences because the tiled advantage no longer applies. The row/column buffer pair measures the cost of transposing a flat layout that the images could recover by tiling.

## 5. Experimental Design
### Independent variables
- Representation: row-major storage buffer, transposed (column-major) storage buffer, R32_UINT storage image (GENERAL layout), R32_UINT sampled image via usampler2D texelFetch (SHADER_READ_ONLY_OPTIMAL).
- Access mode per invocation: neighborhood clamped (the invocation's 2x2 foot) or hashed random scatter. Four exact fetches per invocation in every combination.
- Problem size: 64 Ki to the budget/device cap (a multiple of the 1024-wide grid), W x H defined per size. Both image axes must fit maxImageDimension2D; height caps the selected count even when the buffer budget and dispatch limit would permit more elements.

### Controlled variables
- Coordinates, fetch count (4), mixing arithmetic, output position, and output count are identical across representations at a given access mode.
- Dataset content is coordinate-hash only; no filtering, no format conversion, one mipmap, one sample.
- All upload and layout-transition work is staged once per (size, representation) outside the timed dispatch and exported separately.

### Metrics
- Dispatch GPU ms per case; GB/s over `estimated_shader_request_bytes` = 5 words x element count (four fetches plus one write), identical across variants.
- Representation ratios vs the row-major row-major baseline per access mode; per-iteration FNV-64 checksums; p95/median stability.
- Image footprint notes: allocated image bytes, memory type/heap, and the measured upload and transition times for image representations.

## 6. Implementation Plan
1. Verify R32_UINT supports storage, sampled, and transfer usage on the device; fail clearly otherwise.
2. Create row/column buffers plus storage and sampled images; upload once per size and representation with measured transition stages.
3. Dispatch one shared shader family per representation (access mode via push constant).
4. Validate every timed iteration against the CPU oracle and record checksums.
5. Export request-byte basis, image footprints, and timing-scope notes with every row.

## 7. Analysis Prompts
- Does the storage image's implemented tiling beat row-major under neighborhood fetches, and by how much at each working set?
- Do image and buffer paths converge under random scatter?
- What does the column-buffer penalty say about when engines should keep spreadsheets of transposed copies versus sampling images?
- Which crossover would justify a layout-transition cost in an application? Combine the exported transition times with the dispatch-time savings per reuse group.

## 8. Deliverables
Representation ratio tables by access mode, ms-by-size overview, stability overview, limitations section covering single-format scope, layout-transition bookkeeping, and the absence of per-format sampling paths (float textures, filtering, compression).

Minimum artifact set:
- one core chart (representation ratios)
- one summary table plus representation-effects table
- one short conclusions page with limitations (`results.md`)

## 9. Portfolio Framing Notes
- Frame as: "I can compare buffer and image memory paths under identical logical access and say when 2D tiling, staging, and layout transitions pay."
- Do not generalize from R32_UINT to float or compressed formats; that axis is future work.
- The image path is the first in this suite; its upload/transition numbers quantify costs the buffer-only experiments never paid.
