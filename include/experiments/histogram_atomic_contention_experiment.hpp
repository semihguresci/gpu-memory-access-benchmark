#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct HistogramAtomicContentionExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string global_shader_path;
    std::string privatized_shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0xC0FFEE23U;
};

struct HistogramAtomicContentionExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

HistogramAtomicContentionExperimentOutput
run_histogram_atomic_contention_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                           const HistogramAtomicContentionExperimentConfig& config);
