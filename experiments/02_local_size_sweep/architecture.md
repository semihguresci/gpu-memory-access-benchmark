# Experiment 02 Architecture

## 1. Purpose
Experiment 02 extends the baseline harness with a controlled sweep over legal `local_size_x` values while keeping kernel logic, memory layout, and synchronization behavior fixed.

Primary architectural goals:
- isolate local-size effects without changing algorithm semantics
- keep correctness checks mandatory for every measured row
- preserve explicit Vulkan ownership and teardown rules
- produce reproducible row-level data for post-run analysis

## 2. Runtime Component Architecture
```mermaid
flowchart LR
    CLI["CLI Options<br/>--experiment --size --warmup --iterations"] --> MAIN["main.cpp"]
    MAIN --> REG["Generated Experiment Registry"]
    REG --> ADP["local_size_sweep_adapter.cpp"]
    ADP --> EXP["LocalSizeSweepExperiment"]

    EXP --> CAPS["Device Capability Probe<br/>workgroup size/invocations/count"]
    EXP --> SHD["Shader Variant Set<br/>02_local_size_*.comp.spv"]
    EXP --> VCU["VulkanComputeUtils<br/>pipeline/descriptor helpers"]
    EXP --> BUF["BufferUtils<br/>device + staging buffers"]
    EXP --> TIM["VulkanContext::measure_gpu_time_ms"]

    EXP --> ROWS["BenchmarkMeasurementRow[]"]
    EXP --> SUM["BenchmarkResult[]"]
    MAIN --> JSON["JsonExporter"]
    ROWS --> JSON
    SUM --> JSON
    JSON --> OUT["benchmark_results.json"]
```

## 3. Resource Ownership Model
Per active local-size variant:
- shader module
- descriptor set layout
- descriptor pool + descriptor set
- pipeline layout
- compute pipeline

Shared across variants:
- device-local storage buffer
- upload staging buffer
- readback staging buffer

Ownership rule:
- experiment function creates and destroys resources
- destruction runs in strict reverse creation order
- all destroyed Vulkan handles are reset to `VK_NULL_HANDLE`

## 4. Sweep Execution Flow
```mermaid
flowchart TD
    A["Build candidate local_size list"] --> B["Filter by device limits"]
    B --> C{"Any legal sizes?"}
    C -- No --> Z["Fail run with actionable error"]
    C -- Yes --> D["Build problem-size sweep (clamped)"]
    D --> E["For each legal local_size"]
    E --> F["Create/lookup matching pipeline"]
    F --> G["For each problem_size"]
    G --> H["Warmup iterations"]
    H --> I["Timed iterations"]
    I --> J["Upload -> Dispatch -> Readback"]
    J --> K["Validate correctness"]
    K --> L["Append row metrics + notes"]
    L --> M["Summarize samples (median/p95)"]
    M --> N["Export JSON for analysis scripts"]
```

## 5. Per-Iteration Command Sequence
```mermaid
sequenceDiagram
    participant CPU as "Host Thread"
    participant GPU as "GPU Queue"
    participant UP as "Upload Stage"
    participant DP as "Dispatch Stage (local_size = N)"
    participant RB as "Readback Stage"

    CPU->>CPU: "Fill upload staging sentinel"
    CPU->>UP: "measure_gpu_time_ms(upload)"
    UP->>GPU: "vkCmdCopyBuffer(upload -> device)"
    UP->>GPU: "transfer-write -> compute-read/write barrier"
    GPU-->>UP: "upload_ms"

    CPU->>DP: "measure_gpu_time_ms(dispatch)"
    DP->>GPU: "vkCmdBindPipeline + vkCmdBindDescriptorSets"
    DP->>GPU: "vkCmdDispatch(group_count_x, 1, 1) repeated"
    DP->>GPU: "compute-write -> transfer-read barrier"
    GPU-->>DP: "dispatch_ms"

    CPU->>RB: "measure_gpu_time_ms(readback)"
    RB->>GPU: "vkCmdCopyBuffer(device -> readback)"
    GPU-->>RB: "readback_ms"

    CPU->>CPU: "Validate readback and append BenchmarkMeasurementRow"
```

## 6. Data and Analysis Pipeline
```mermaid
flowchart LR
    RUN["gpu_memory_layout_experiments --experiment 02_local_size_sweep"] --> RAW["results/tables/benchmark_results.json"]
    RAW --> ANALYZE["analyze_local_size_sweep.py"]
    ANALYZE --> PIVOT["results/tables/local_size_pivot.csv"]
    ANALYZE --> RANK["results/tables/local_size_ranking.csv"]
    RAW --> PLOT["plot_results.py"]
    PLOT --> CHART1["results/charts/local_size_vs_gpu_ms.png"]
    PLOT --> CHART2["results/charts/local_size_vs_throughput.png"]
    PIVOT --> RES["results.md"]
    RANK --> RES
    CHART1 --> RES
    CHART2 --> RES
```

## 7. Key Architectural Constraints
- Candidate local sizes must be filtered before dispatch recording.
- Synchronization remains explicit and local to each phase.
- Correctness is a hard gate for interpreting performance data.
- Result encoding must preserve enough metadata to compare local sizes reliably.
