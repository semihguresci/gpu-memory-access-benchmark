#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct StreamCompactionExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string atomic_append_shader_path;
    std::string stage1_shader_path;
    std::string stage2_shader_path;
    std::string stage3_shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x240C0A24U;
};

struct StreamCompactionExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

StreamCompactionExperimentOutput run_stream_compaction_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                                  const StreamCompactionExperimentConfig& config);
