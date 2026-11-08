#pragma once

#include "benchmark_runner.hpp"

#include <string>
#include <vector>

struct BenchmarkExportMetadata {
    std::string gpu_name;
    std::string vulkan_api_version;
    std::string driver_version;
    std::string gpu_device_uuid;
    bool validation_enabled = false;
    bool gpu_timestamps_supported = false;
    int warmup_iterations = 0;
    int timed_iterations = 0;
    std::string measurement_revision = "2026-11-08-contention-merge";
    std::vector<std::string> command_line;
    std::string shader_binary_directory;
};

class JsonExporter {
  public:
    static constexpr const char* kSchemaName = "gpu_memory_layout_results";
    static constexpr const char* kSchemaVersion = "1.2.0";

    static void write_benchmark_results(const std::vector<BenchmarkResult>& results, const std::string& output_path);
    static void write_benchmark_results(const std::vector<BenchmarkResult>& results,
                                        const std::vector<BenchmarkMeasurementRow>& rows,
                                        const BenchmarkExportMetadata& metadata, const std::string& output_path);
};
