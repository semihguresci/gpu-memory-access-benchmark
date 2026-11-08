#pragma once

#include "benchmark_runner.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

struct AppOptions;
class VulkanContext;

struct ExperimentRunOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool success = true;
    std::string error_message;
};

using ExperimentRunFn = bool (*)(VulkanContext& context, const BenchmarkRunner& runner, const AppOptions& options,
                                 ExperimentRunOutput& output);

struct ExperimentDescriptor {
    std::string_view id;
    std::string_view display_name;
    std::string_view category;
    bool enabled = false;
    ExperimentRunFn run = nullptr;
    std::array<uint32_t, 3> required_local_size{1U, 1U, 1U};
    uint32_t required_shared_memory_bytes = 0;
};
