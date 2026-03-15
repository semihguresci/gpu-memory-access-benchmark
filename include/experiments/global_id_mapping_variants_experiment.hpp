#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct GlobalIdMappingVariantsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct GlobalIdMappingVariantsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

GlobalIdMappingVariantsExperimentOutput
run_global_id_mapping_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                          const GlobalIdMappingVariantsExperimentConfig& config);
