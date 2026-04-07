#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct DeviceLocalVsHostVisibleHeapPlacementExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct DeviceLocalVsHostVisibleHeapPlacementExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

DeviceLocalVsHostVisibleHeapPlacementExperimentOutput run_device_local_vs_host_visible_heap_placement_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const DeviceLocalVsHostVisibleHeapPlacementExperimentConfig& config);
