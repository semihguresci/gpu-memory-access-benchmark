#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct SubgroupStreamCompactionVariantsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shared_atomic_shader_path;
    std::string subgroup_ballot_shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x32C0A241U;
};

struct SubgroupStreamCompactionVariantsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SubgroupStreamCompactionVariantsExperimentOutput
run_subgroup_stream_compaction_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                   const SubgroupStreamCompactionVariantsExperimentConfig& config);
