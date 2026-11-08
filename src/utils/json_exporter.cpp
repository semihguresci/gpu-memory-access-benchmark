#include "utils/json_exporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

std::string utc_now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto whole_seconds = std::chrono::floor<std::chrono::seconds>(now);
    const auto fractional_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - whole_seconds);
    const std::time_t now_time = std::chrono::system_clock::to_time_t(whole_seconds);

    std::tm utc_tm{};
#ifdef _WIN32
    gmtime_s(&utc_tm, &now_time);
#else
    gmtime_r(&now_time, &utc_tm);
#endif

    std::ostringstream stream;
    stream << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(6)
           << fractional_microseconds.count() << 'Z';
    return stream.str();
}

} // namespace

void JsonExporter::write_benchmark_results(const std::vector<BenchmarkResult>& results,
                                           const std::string& output_path) {
    write_benchmark_results(results, {}, BenchmarkExportMetadata{}, output_path);
}

void JsonExporter::write_benchmark_results(const std::vector<BenchmarkResult>& results,
                                           const std::vector<BenchmarkMeasurementRow>& rows,
                                           const BenchmarkExportMetadata& metadata, const std::string& output_path) {
    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    nlohmann::json root = nlohmann::json::object();
    root["schema"] = {{"name", kSchemaName}, {"version", kSchemaVersion}};
    root["metadata"] = {{"exported_at_utc", utc_now_iso8601()},
                        {"generator", "gpu_memory_layout_experiments"},
                        {"result_count", results.size()},
                        {"row_count", rows.size()},
                        {"gpu_name", metadata.gpu_name},
                        {"vulkan_api_version", metadata.vulkan_api_version},
                        {"driver_version", metadata.driver_version},
                        {"gpu_device_uuid", metadata.gpu_device_uuid},
                        {"validation_enabled", metadata.validation_enabled},
                        {"gpu_timestamps_supported", metadata.gpu_timestamps_supported},
                        {"warmup_iterations", metadata.warmup_iterations},
                        {"timed_iterations", metadata.timed_iterations},
                        {"measurement_revision", metadata.measurement_revision},
                        {"command_line", metadata.command_line},
                        {"shader_binary_directory", metadata.shader_binary_directory}};
    root["results"] = nlohmann::json::array();

    for (const auto& result : results) {
        root["results"].push_back({{"experiment", result.experiment_name},
                                   {"average_ms", result.average_ms},
                                   {"min_ms", result.min_ms},
                                   {"max_ms", result.max_ms},
                                   {"median_ms", result.median_ms},
                                   {"p95_ms", result.p95_ms},
                                   {"sample_count", result.sample_count}});
    }

    if (!rows.empty()) {
        root["rows"] = nlohmann::json::array();
        for (const auto& row : rows) {
            root["rows"].push_back({{"experiment_id", row.experiment_id},
                                    {"variant", row.variant},
                                    {"problem_size", row.problem_size},
                                    {"dispatch_count", row.dispatch_count},
                                    {"iteration", row.iteration},
                                    {"gpu_ms", row.gpu_ms},
                                    {"end_to_end_ms", row.end_to_end_ms},
                                    {"throughput", row.throughput},
                                    {"gbps", row.gbps},
                                    {"correctness_pass", row.correctness_pass},
                                    {"notes", row.notes}});
        }
    }

    std::ofstream output(output_path);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open benchmark export for writing: " + output_path);
    }
    output << root.dump(2) << "\n";
    output.flush();
    if (!output.good()) {
        throw std::runtime_error("Failed to write benchmark export: " + output_path);
    }
    output.close();
    if (output.fail()) {
        throw std::runtime_error("Failed to close benchmark export: " + output_path);
    }
}
