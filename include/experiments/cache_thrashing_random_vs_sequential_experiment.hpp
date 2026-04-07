#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct CacheThrashingRandomVsSequentialExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x6C8E9CF5U;
    std::uint32_t block_size = 256U;
};

struct CacheThrashingRandomVsSequentialExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

CacheThrashingRandomVsSequentialExperimentOutput
run_cache_thrashing_random_vs_sequential_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const CacheThrashingRandomVsSequentialExperimentConfig& config);
