#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SubgroupOperationsStudyExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct SubgroupOperationsStudyExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SubgroupOperationsStudyExperimentOutput
run_subgroup_operations_study_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                         const SubgroupOperationsStudyExperimentConfig& config);
