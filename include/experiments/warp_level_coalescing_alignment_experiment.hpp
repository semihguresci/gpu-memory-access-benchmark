#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct WarpLevelCoalescingAlignmentExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct WarpLevelCoalescingAlignmentExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

WarpLevelCoalescingAlignmentExperimentOutput
run_warp_level_coalescing_alignment_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                               const WarpLevelCoalescingAlignmentExperimentConfig& config);
