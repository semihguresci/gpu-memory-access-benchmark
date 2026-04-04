#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct ScatterAccessPatternExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x51A7C3D9U;
    std::uint32_t collision_factor = 4U;
    std::uint32_t hot_window_size = 32U;
};

struct ScatterAccessPatternExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

ScatterAccessPatternExperimentOutput
run_scatter_access_pattern_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const ScatterAccessPatternExperimentConfig& config);
