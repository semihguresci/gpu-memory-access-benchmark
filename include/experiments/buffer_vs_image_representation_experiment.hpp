#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct BufferVsImageRepresentationExperimentConfig {
    // Per-buffer budget for the seven live allocations (row buffer, column
    // buffer, storage image, sampled image, output buffer, staging in, staging out).
    std::size_t max_buffer_bytes = 0;
    // Original CLI scratch total, exported separately from the per-buffer budget.
    std::size_t scratch_size_bytes = 0;
    std::string row_buffer_shader_path;
    std::string column_buffer_shader_path;
    std::string storage_image_shader_path;
    std::string sampled_image_shader_path;
    bool verbose_progress = false;
};

struct BufferVsImageRepresentationExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

BufferVsImageRepresentationExperimentOutput
run_buffer_vs_image_representation_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                              const BufferVsImageRepresentationExperimentConfig& config);
