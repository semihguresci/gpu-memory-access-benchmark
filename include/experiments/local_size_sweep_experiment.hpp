#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class VulkanContext;

struct LocalSizeSweepExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    bool include_noop_variant = true;
    uint32_t dispatch_count = 1;
};

struct LocalSizeSweepExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

LocalSizeSweepExperimentOutput run_local_size_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                               const LocalSizeSweepExperimentConfig& config);
