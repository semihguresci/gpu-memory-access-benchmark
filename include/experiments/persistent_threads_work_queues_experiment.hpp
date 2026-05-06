#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VulkanContext;

struct PersistentThreadsWorkQueuesExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
    std::uint32_t pattern_seed = 0x40101013U;
};

struct PersistentThreadsWorkQueuesExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

PersistentThreadsWorkQueuesExperimentOutput
run_persistent_threads_work_queues_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                              const PersistentThreadsWorkQueuesExperimentConfig& config);
