#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct DispatchBasicsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string write_shader_path;
    std::string noop_shader_path;
    bool include_noop_variant = true;
    bool verbose_progress = false;
};

struct DispatchBasicsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

DispatchBasicsExperimentOutput run_dispatch_basics_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                              const DispatchBasicsExperimentConfig& config);
