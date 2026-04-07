#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct BarrierSynchronizationCostExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string flat_shader_path;
    std::string tiled_shader_path;
    bool verbose_progress = false;
};

struct BarrierSynchronizationCostExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

BarrierSynchronizationCostExperimentOutput
run_barrier_synchronization_cost_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                            const BarrierSynchronizationCostExperimentConfig& config);
