#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct TiledLightAssignmentExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x390D0D12U;
};

struct TiledLightAssignmentExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

TiledLightAssignmentExperimentOutput
run_tiled_light_assignment_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const TiledLightAssignmentExperimentConfig& config);
