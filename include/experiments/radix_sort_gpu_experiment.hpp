#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct RadixSortGpuExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string count_shader_path;
    std::string scan_shader_path;
    std::string scatter_shader_path;
    bool verbose_progress = false;
};

struct RadixSortGpuExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

RadixSortGpuExperimentOutput run_radix_sort_gpu_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                           const RadixSortGpuExperimentConfig& config);
