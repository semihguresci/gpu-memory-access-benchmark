#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct AosoaBlockedLayoutExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string aos_shader_path;
    std::string soa_shader_path;
    std::string aosoa_shader_path;
    bool verbose_progress = false;
};

struct AosoaBlockedLayoutExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

AosoaBlockedLayoutExperimentOutput
run_aosoa_blocked_layout_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const AosoaBlockedLayoutExperimentConfig& config);
