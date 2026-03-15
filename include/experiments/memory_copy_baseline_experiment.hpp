#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct MemoryCopyBaselineExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string read_only_shader_path;
    std::string write_only_shader_path;
    std::string read_write_copy_shader_path;
    bool verbose_progress = false;
};

struct MemoryCopyBaselineExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

MemoryCopyBaselineExperimentOutput
run_memory_copy_baseline_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const MemoryCopyBaselineExperimentConfig& config);
