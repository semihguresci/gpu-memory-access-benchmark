#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct SpatialBinningClusteredCullingCapstoneExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string append_shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x250B105U;
};

struct SpatialBinningClusteredCullingCapstoneExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SpatialBinningClusteredCullingCapstoneExperimentOutput run_spatial_binning_clustered_culling_capstone_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const SpatialBinningClusteredCullingCapstoneExperimentConfig& config);
