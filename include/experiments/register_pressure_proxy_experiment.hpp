#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct RegisterPressureProxyExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct RegisterPressureProxyExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

RegisterPressureProxyExperimentOutput
run_register_pressure_proxy_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const RegisterPressureProxyExperimentConfig& config);
