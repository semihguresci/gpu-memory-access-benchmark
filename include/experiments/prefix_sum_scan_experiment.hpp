#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct PrefixSumScanExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string scan_shader_path;
    std::string add_offsets_shader_path;
    bool verbose_progress = false;
};

struct PrefixSumScanExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

PrefixSumScanExperimentOutput run_prefix_sum_scan_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                             const PrefixSumScanExperimentConfig& config);
