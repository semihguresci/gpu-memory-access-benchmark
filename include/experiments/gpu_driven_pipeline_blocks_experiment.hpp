#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct GpuDrivenPipelineBlocksExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct GpuDrivenPipelineBlocksExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

GpuDrivenPipelineBlocksExperimentOutput
run_gpu_driven_pipeline_blocks_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                          const GpuDrivenPipelineBlocksExperimentConfig& config);
