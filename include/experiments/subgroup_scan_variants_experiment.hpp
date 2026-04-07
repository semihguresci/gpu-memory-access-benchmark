#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SubgroupScanVariantsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shared_scan_shader_path;
    std::string subgroup_scan_shader_path;
    bool verbose_progress = false;
};

struct SubgroupScanVariantsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SubgroupScanVariantsExperimentOutput
run_subgroup_scan_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const SubgroupScanVariantsExperimentConfig& config);
