#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SubgroupReductionVariantsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shared_tree_shader_path;
    std::string subgroup_hybrid_shader_path;
    bool verbose_progress = false;
};

struct SubgroupReductionVariantsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SubgroupReductionVariantsExperimentOutput
run_subgroup_reduction_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                           const SubgroupReductionVariantsExperimentConfig& config);
