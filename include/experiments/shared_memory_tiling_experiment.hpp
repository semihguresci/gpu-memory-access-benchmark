#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SharedMemoryTilingExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string direct_shader_path;
    std::string tiled_shader_path;
    bool verbose_progress = false;
};

struct SharedMemoryTilingExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SharedMemoryTilingExperimentOutput
run_shared_memory_tiling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const SharedMemoryTilingExperimentConfig& config);
