#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct GatherAccessPatternExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x6C8E9CF5U;
    std::uint32_t block_size = 32U;
    std::uint32_t cluster_size = 256U;
};

struct GatherAccessPatternExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

GatherAccessPatternExperimentOutput
run_gather_access_pattern_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                     const GatherAccessPatternExperimentConfig& config);
