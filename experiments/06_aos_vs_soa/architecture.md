# Experiment 06 Architecture

## 1. Purpose
Experiment 06 compares two memory layouts for the same particle-update kernel:
- `aos`: array of packed structs
- `soa`: structure of arrays (8 bound float buffers)

Both variants execute the same arithmetic update to isolate layout effects.

## 2. Runtime Component Architecture
```mermaid
flowchart LR
    CLI["CLI Options<br/>--experiment --size --warmup --iterations"] --> MAIN["main.cpp"]
    MAIN --> REG["Generated Experiment Registry"]
    REG --> ADP["aos_soa_adapter.cpp"]
    ADP --> EXP["AosSoaExperiment"]

    EXP --> SHD["06_aos.comp.spv / 06_soa.comp.spv"]
    EXP --> BUF["Host-visible storage buffers"]
    EXP --> TIM["VulkanContext::measure_gpu_time_ms"]
    EXP --> ROWS["BenchmarkMeasurementRow[]"]
    EXP --> SUM["BenchmarkResult[]"]
    MAIN --> JSON["JsonExporter"]
    ROWS --> JSON
    SUM --> JSON
```

## 3. Ownership Model
- Experiment runtime owns all Vulkan objects created for AoS and SoA paths.
- Teardown is reverse-order and explicit.
- Handles are reset to `VK_NULL_HANDLE` after destruction.

## 4. Execution Flow
```mermaid
flowchart TD
    A["Resolve shaders + select particle counts"] --> B["Create AoS resources"]
    B --> C["Create SoA resources"]
    C --> D["For each particle count"]
    D --> E["Run AoS warmup/timed loop"]
    E --> F["Run SoA warmup/timed loop"]
    F --> G["Append rows + summaries"]
    G --> H["Destroy AoS/SoA resources"]
```

## 5. Per-Iteration Logic
1. Fill deterministic seed values.
2. Dispatch one compute workload timed with GPU timestamps.
3. Validate output against CPU expected values.
4. Emit row metrics and notes.
