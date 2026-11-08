#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct DependentLoadLatencyExperimentConfig {
    // Per-buffer budget (pointer table, output, upload staging, readback staging).
    std::size_t max_buffer_bytes = 0;
    // Original CLI scratch total, exported separately from the per-buffer budget.
    std::size_t scratch_size_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct DependentLoadLatencyExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

DependentLoadLatencyExperimentOutput
run_dependent_load_latency_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const DependentLoadLatencyExperimentConfig& config);
