#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct BlockReorderFactorialExperimentConfig {
    // Per-buffer upper bound for source, output, block permutation, and staging.
    // The total budget first reserves the 64 MiB eviction buffer; the remaining
    // bytes are split across those five workload/staging buffers.
    std::size_t max_buffer_bytes = 0;
    // Total CLI payload budget for all six live buffers, including eviction.
    std::size_t scratch_size_bytes = 0;
    // Block widths are 2/4/8/16/32 times the queried subgroup size. Sizes are
    // rounded to complete workgroups and complete blocks for every variant.
    std::string shader_path;
    std::string sweep_shader_path;
    bool verbose_progress = false;
};

struct BlockReorderFactorialExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

BlockReorderFactorialExperimentOutput
run_block_reorder_factorial_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const BlockReorderFactorialExperimentConfig& config);
