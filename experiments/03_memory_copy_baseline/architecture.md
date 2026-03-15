# Experiment 03 Architecture

## 1. Purpose
Experiment 03 measures raw memory-path behavior with tightly scoped kernels:
- read-only memory traffic
- write-only memory traffic
- read+write copy traffic

The architecture keeps synchronization explicit and correctness mandatory before interpreting throughput.

## 2. Runtime Component Architecture
```mermaid
flowchart LR
    CLI["CLI Options<br/>--experiment --size --warmup --iterations"] --> MAIN["main.cpp"]
    MAIN --> REG["Generated Experiment Registry"]
    REG --> ADP["memory_copy_baseline_adapter.cpp"]
    ADP --> EXP["MemoryCopyBaselineExperiment"]

    EXP --> BUF["BufferUtils<br/>src/dst/staging buffers"]
    EXP --> SHD["03_memory_*.comp.spv shaders"]
    EXP --> VCU["VulkanComputeUtils<br/>pipeline + barriers"]
    EXP --> TIM["VulkanContext::measure_gpu_time_ms"]
    EXP --> ROWS["BenchmarkMeasurementRow[]"]
    EXP --> SUM["BenchmarkResult[]"]
    MAIN --> JSON["JsonExporter"]
    ROWS --> JSON
    SUM --> JSON
```

## 3. Resource Ownership Model
Shared buffers:
- `src_device` (device-local storage + transfer)
- `dst_device` (device-local storage + transfer)
- `staging` (host-visible transfer src/dst)

Per-mode pipeline resources:
- shader module
- descriptor set layout
- descriptor pool + descriptor set
- pipeline layout
- compute pipeline

Ownership rule:
- experiment function creates and destroys all resources
- teardown is reverse-order
- handles are reset to `VK_NULL_HANDLE`

## 4. Execution Flow
```mermaid
flowchart TD
    A["Resolve shaders + clamp sweep sizes"] --> B["Create buffers + pipelines"]
    B --> C["Map staging memory"]
    C --> D["For each problem size"]
    D --> E["For each mode (read/write/copy)"]
    E --> F["Warmup iterations"]
    F --> G["Timed iterations"]
    G --> H["Upload stage"]
    H --> I["Dispatch stage"]
    I --> J["Readback stage"]
    J --> K["Validate correctness"]
    K --> L["Append row + notes"]
    L --> M["Summarize dispatch samples"]
    M --> N["Unmap + destroy resources"]
```

## 5. Per-Iteration Command Sequence
```mermaid
sequenceDiagram
    participant CPU as "Host Thread"
    participant GPU as "GPU Queue"

    CPU->>CPU: "Prepare staging payload"
    CPU->>GPU: "Upload copy (staging -> device)"
    CPU->>GPU: "Dispatch selected mode"
    CPU->>GPU: "Readback copy (device -> staging)"
    CPU->>CPU: "Validate expected contents"
    CPU->>CPU: "Record row (gpu_ms/end_to_end/gbps)"
```

## 6. Data and Analysis Pipeline
```mermaid
flowchart LR
    RUN["run_experiment_data_collection.py --experiment 03_memory_copy_baseline"] --> RAW["benchmark_results.json"]
    RAW --> COLLECT["scripts/collect_run.py"]
    RAW --> ANALYZE["scripts/analyze_memory_copy_baseline.py"]
    RAW --> PLOT["scripts/plot_results.py"]
    ANALYZE --> TABLES["results/tables/*.csv"]
    ANALYZE --> CHARTS["results/charts/*.png"]
```
