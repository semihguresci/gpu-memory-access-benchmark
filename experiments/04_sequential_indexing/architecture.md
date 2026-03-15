# Experiment 04 Architecture

## 1. Purpose
Experiment 04 measures contiguous thread-to-data mapping in a simple read+write kernel:
- source read at index `i`
- destination write at index `i`
- deterministic transform for correctness validation

This experiment establishes a strong baseline before non-sequential mapping studies.

## 2. Runtime Component Architecture
```mermaid
flowchart LR
    CLI["CLI Options<br/>--experiment --size --warmup --iterations"] --> MAIN["main.cpp"]
    MAIN --> REG["Generated Experiment Registry"]
    REG --> ADP["sequential_indexing_adapter.cpp"]
    ADP --> EXP["SequentialIndexingExperiment"]

    EXP --> BUF["BufferUtils<br/>src/dst/staging buffers"]
    EXP --> SHD["04_sequential_indexing.comp.spv shader"]
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

Pipeline resources:
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
    A["Resolve shader + clamp sweep sizes"] --> B["Create buffers + pipeline"]
    B --> C["Map staging memory"]
    C --> D["For each problem size"]
    D --> E["For each dispatch count"]
    E --> F["Warmup iterations"]
    F --> G["Timed iterations"]
    G --> H["Upload src"]
    H --> I["Upload dst sentinel"]
    I --> J["Dispatch sequential kernel"]
    J --> K["Readback dst"]
    K --> L["Validate correctness"]
    L --> M["Append row + notes"]
    M --> N["Summarize dispatch samples"]
    N --> O["Unmap + destroy resources"]
```

## 5. Per-Iteration Command Sequence
```mermaid
sequenceDiagram
    participant CPU as "Host Thread"
    participant GPU as "GPU Queue"

    CPU->>CPU: "Prepare source and destination staging payload"
    CPU->>GPU: "Upload source (staging -> src_device)"
    CPU->>GPU: "Upload destination sentinel (staging -> dst_device)"
    CPU->>GPU: "Dispatch sequential read+write kernel"
    CPU->>GPU: "Readback destination (dst_device -> staging)"
    CPU->>CPU: "Validate expected contents"
    CPU->>CPU: "Record row (gpu_ms/end_to_end/gbps)"
```

## 6. Data and Analysis Pipeline
```mermaid
flowchart LR
    RUN["run_experiment_data_collection.py --experiment 04_sequential_indexing"] --> RAW["benchmark_results.json"]
    RAW --> COLLECT["scripts/collect_run.py"]
    RAW --> ANALYZE["scripts/analyze_sequential_indexing.py"]
    RAW --> PLOT["scripts/plot_results.py"]
    ANALYZE --> TABLES["results/tables/*.csv"]
    ANALYZE --> CHARTS["results/charts/*.png"]
```
