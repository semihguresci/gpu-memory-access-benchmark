#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct TwoDimensionalLocalityTransposeStudyExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string row_major_copy_shader_path;
    std::string naive_transpose_shader_path;
    std::string tiled_transpose_shader_path;
    std::string tiled_transpose_padded_shader_path;
    bool verbose_progress = false;
};

struct TwoDimensionalLocalityTransposeStudyExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

TwoDimensionalLocalityTransposeStudyExperimentOutput run_two_dimensional_locality_transpose_study_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const TwoDimensionalLocalityTransposeStudyExperimentConfig& config);
