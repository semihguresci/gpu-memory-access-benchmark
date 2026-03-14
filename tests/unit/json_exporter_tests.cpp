#include "support/filesystem_test_utils.hpp"
#include "utils/json_exporter.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

nlohmann::json load_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    EXPECT_TRUE(input.is_open());
    return nlohmann::json::parse(input);
}

} // namespace

TEST(JsonExporterTests, WritesSummaryOnlySchemaAndMetadata) {
    TestSupport::ScopedTempDirectory temp_dir;
    const std::filesystem::path output_path = temp_dir.path() / "nested" / "summary.json";

    const std::vector<BenchmarkResult> results = {
        BenchmarkResult{.experiment_name = "01_dispatch_basics",
                        .average_ms = 1.5,
                        .min_ms = 1.0,
                        .max_ms = 2.0,
                        .median_ms = 1.4,
                        .p95_ms = 1.9,
                        .sample_count = 10},
    };

    JsonExporter::write_benchmark_results(results, output_path.string());

    ASSERT_TRUE(std::filesystem::exists(output_path));
    const nlohmann::json root = load_json(output_path);

    ASSERT_TRUE(root.contains("schema"));
    EXPECT_EQ(root["schema"]["name"], JsonExporter::kSchemaName);
    EXPECT_EQ(root["schema"]["version"], JsonExporter::kSchemaVersion);

    ASSERT_TRUE(root.contains("metadata"));
    EXPECT_EQ(root["metadata"]["result_count"], 1);
    EXPECT_EQ(root["metadata"]["row_count"], 0);
    EXPECT_TRUE(root["metadata"]["exported_at_utc"].is_string());

    ASSERT_TRUE(root.contains("results"));
    ASSERT_EQ(root["results"].size(), 1U);
    EXPECT_EQ(root["results"][0]["experiment"], "01_dispatch_basics");
    EXPECT_FALSE(root.contains("rows"));
}

TEST(JsonExporterTests, WritesRowsAndMetadataFields) {
    TestSupport::ScopedTempDirectory temp_dir;
    const std::filesystem::path output_path = temp_dir.path() / "full.json";

    const std::vector<BenchmarkResult> results = {
        BenchmarkResult{.experiment_name = "06_aos_particles_131072",
                        .average_ms = 0.8,
                        .min_ms = 0.6,
                        .max_ms = 1.1,
                        .median_ms = 0.75,
                        .p95_ms = 1.0,
                        .sample_count = 25},
    };

    const std::vector<BenchmarkMeasurementRow> rows = {
        BenchmarkMeasurementRow{.experiment_id = "06_aos_soa",
                                .variant = "aos",
                                .problem_size = 131072,
                                .dispatch_count = 1,
                                .iteration = 0,
                                .gpu_ms = 0.7,
                                .end_to_end_ms = 0.9,
                                .throughput = 1.2e8,
                                .gbps = 2.4,
                                .correctness_pass = true,
                                .notes = "ok"},
    };

    const BenchmarkExportMetadata metadata{
        .gpu_name = "Test GPU",
        .vulkan_api_version = "1.3.280",
        .driver_version = "999.1",
        .validation_enabled = true,
        .gpu_timestamps_supported = true,
        .warmup_iterations = 3,
        .timed_iterations = 10,
    };

    JsonExporter::write_benchmark_results(results, rows, metadata, output_path.string());

    ASSERT_TRUE(std::filesystem::exists(output_path));
    const nlohmann::json root = load_json(output_path);

    ASSERT_TRUE(root.contains("metadata"));
    EXPECT_EQ(root["metadata"]["gpu_name"], "Test GPU");
    EXPECT_EQ(root["metadata"]["vulkan_api_version"], "1.3.280");
    EXPECT_EQ(root["metadata"]["driver_version"], "999.1");
    EXPECT_EQ(root["metadata"]["validation_enabled"], true);
    EXPECT_EQ(root["metadata"]["gpu_timestamps_supported"], true);
    EXPECT_EQ(root["metadata"]["warmup_iterations"], 3);
    EXPECT_EQ(root["metadata"]["timed_iterations"], 10);
    EXPECT_EQ(root["metadata"]["row_count"], 1);

    ASSERT_TRUE(root.contains("rows"));
    ASSERT_EQ(root["rows"].size(), 1U);
    EXPECT_EQ(root["rows"][0]["experiment_id"], "06_aos_soa");
    EXPECT_EQ(root["rows"][0]["variant"], "aos");
    EXPECT_EQ(root["rows"][0]["correctness_pass"], true);
}
