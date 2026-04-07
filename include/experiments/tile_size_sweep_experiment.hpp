#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct TileSizeSweepExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string direct_shader_path;
    std::string tiled_shader_path;
    bool verbose_progress = false;
};

struct TileSizeSweepExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

TileSizeSweepExperimentOutput run_tile_size_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                             const TileSizeSweepExperimentConfig& config);
