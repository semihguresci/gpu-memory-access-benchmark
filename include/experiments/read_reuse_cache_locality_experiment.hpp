#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct ReadReuseCacheLocalityExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct ReadReuseCacheLocalityExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

ReadReuseCacheLocalityExperimentOutput
run_read_reuse_cache_locality_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                         const ReadReuseCacheLocalityExperimentConfig& config);
