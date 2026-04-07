#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct Vec3Vec4PaddingCostsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string vec3_shader_path;
    std::string vec4_shader_path;
    std::string split_scalars_shader_path;
    bool verbose_progress = false;
};

struct Vec3Vec4PaddingCostsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

Vec3Vec4PaddingCostsExperimentOutput
run_vec3_vec4_padding_costs_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const Vec3Vec4PaddingCostsExperimentConfig& config);
