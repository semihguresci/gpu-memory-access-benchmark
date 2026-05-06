#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct CrossGpuReproducibilityExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct CrossGpuReproducibilityExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

CrossGpuReproducibilityExperimentOutput
run_cross_gpu_reproducibility_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                         const CrossGpuReproducibilityExperimentConfig& config);
