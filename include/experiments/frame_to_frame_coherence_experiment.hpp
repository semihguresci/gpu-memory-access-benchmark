#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct FrameToFrameCoherenceExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct FrameToFrameCoherenceExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

FrameToFrameCoherenceExperimentOutput
run_frame_to_frame_coherence_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                        const FrameToFrameCoherenceExperimentConfig& config);
