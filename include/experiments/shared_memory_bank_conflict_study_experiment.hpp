#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SharedMemoryBankConflictStudyExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct SharedMemoryBankConflictStudyExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SharedMemoryBankConflictStudyExperimentOutput
run_shared_memory_bank_conflict_study_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                 const SharedMemoryBankConflictStudyExperimentConfig& config);
