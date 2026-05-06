#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct FrustumVsClusteredCullingExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x380C0C11U;
};

struct FrustumVsClusteredCullingExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

FrustumVsClusteredCullingExperimentOutput
run_frustum_vs_clustered_culling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                            const FrustumVsClusteredCullingExperimentConfig& config);
