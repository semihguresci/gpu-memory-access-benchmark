#include "support/filesystem_test_utils.hpp"
#include "vulkan_context.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_DIR
#define GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_DIR ""
#endif

#ifndef GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_NAME
#define GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_NAME ""
#endif

namespace {

struct SmokeCase {
    const char* experiment_id;
    const char* size_arg;
};

// Keep this matrix small but representative so ctest exercises multiple experiment
// families without turning GPU integration coverage into a full benchmark run.
constexpr std::array<SmokeCase, 6> kSmokeCases{{
    {.experiment_id = "01_dispatch_basics", .size_arg = "4M"},
    {.experiment_id = "06_aos_vs_soa", .size_arg = "32M"},
    {.experiment_id = "11_coalesced_vs_strided", .size_arg = "32M"},
    {.experiment_id = "16_shared_memory_tiling", .size_arg = "32M"},
    {.experiment_id = "24_stream_compaction", .size_arg = "32M"},
    {.experiment_id = "27_cache_thrashing_random_vs_sequential", .size_arg = "64M"},
}};

std::filesystem::path benchmark_binary_path() {
    const std::filesystem::path binary_dir = GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_DIR;
    const std::filesystem::path binary_name = GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_NAME;
    return binary_dir / binary_name;
}

std::wstring widen_ascii(const std::string& text) {
    return {text.begin(), text.end()};
}

int run_benchmark_smoke_case(const std::filesystem::path& binary_path, const SmokeCase& smoke_case,
                             const std::filesystem::path& output_base) {
#ifdef _WIN32
    const std::wstring command = L"\"" + binary_path.wstring() + L"\" --experiment " +
                                 widen_ascii(smoke_case.experiment_id) + L" --iterations 1 --warmup 0 --size " +
                                 widen_ascii(smoke_case.size_arg) + L" --output \"" + output_base.wstring() + L"\"";

    std::vector<wchar_t> command_buffer(command.begin(), command.end());
    command_buffer.push_back(L'\0');

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};
    if (CreateProcessW(nullptr, command_buffer.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup_info,
                       &process_info) == FALSE) {
        return -1;
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 1U;
    if (GetExitCodeProcess(process_info.hProcess, &exit_code) == FALSE) {
        exit_code = 1U;
    }

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return static_cast<int>(exit_code);
#else
    const std::string command = "\"" + binary_path.string() + "\" --experiment " + smoke_case.experiment_id +
                                " --iterations 1 --warmup 0 --size " + smoke_case.size_arg + " --output \"" +
                                output_base.string() + "\"";
    return std::system(command.c_str());
#endif
}

} // namespace

TEST(ExperimentSmokeTests, BenchmarkBinaryExistsInIntegrationTarget) {
    const std::filesystem::path binary_path = benchmark_binary_path();
    ASSERT_TRUE(std::filesystem::exists(binary_path)) << "Missing benchmark binary: " << binary_path.string();
}

TEST(ExperimentSmokeTests, ContextIsUsableForSmokeRuns) {
    VulkanContext context;
    if (!context.initialize(false)) {
        GTEST_SKIP() << "VulkanContext initialization failed in this environment.";
    }

    if (!context.gpu_timestamps_supported()) {
        context.shutdown();
        GTEST_SKIP() << "GPU timestamp support is required for experiment smoke tests.";
    }

    context.shutdown();
}

TEST(ExperimentSmokeTests, RunsRepresentativeExperimentMatrix) {
    VulkanContext context;
    if (!context.initialize(false)) {
        GTEST_SKIP() << "VulkanContext initialization failed in this environment.";
    }

    if (!context.gpu_timestamps_supported()) {
        context.shutdown();
        GTEST_SKIP() << "GPU timestamp support is required for experiment smoke tests.";
    }

    context.shutdown();

    const std::filesystem::path binary_path = benchmark_binary_path();
    ASSERT_TRUE(std::filesystem::exists(binary_path)) << "Missing benchmark binary: " << binary_path.string();

    TestSupport::ScopedTempDirectory temp_directory("gpu_memory_layout_experiment_smoke");

    for (const SmokeCase& smoke_case : kSmokeCases) {
        const std::filesystem::path output_base = temp_directory.path() / smoke_case.experiment_id;
        const std::filesystem::path output_json = output_base.string() + ".json";
        // The smoke test validates the exported schema as well as process success so
        // experiment-layer regressions are caught even when the binary still launches.
        const int exit_code = run_benchmark_smoke_case(binary_path, smoke_case, output_base);
        ASSERT_EQ(exit_code, 0) << "Smoke run failed for " << smoke_case.experiment_id
                                << " using binary: " << binary_path.string();
        ASSERT_TRUE(std::filesystem::exists(output_json)) << "Missing smoke output: " << output_json.string();

        std::ifstream file(output_json);
        ASSERT_TRUE(file.is_open()) << "Failed to open smoke output: " << output_json.string();

        nlohmann::json json = nlohmann::json::parse(file, nullptr, true, true);
        ASSERT_TRUE(json.contains("metadata"));
        ASSERT_TRUE(json.contains("rows"));
        ASSERT_TRUE(json.contains("results"));

        const auto& metadata = json.at("metadata");
        const auto& rows = json.at("rows");
        const auto& results = json.at("results");

        ASSERT_TRUE(rows.is_array());
        ASSERT_TRUE(results.is_array());
        EXPECT_EQ(metadata.at("warmup_iterations").get<int>(), 0);
        EXPECT_EQ(metadata.at("timed_iterations").get<int>(), 1);
        EXPECT_EQ(metadata.at("row_count").get<std::size_t>(), rows.size());
        EXPECT_EQ(metadata.at("result_count").get<std::size_t>(), results.size());
        EXPECT_GT(rows.size(), 0U);
        EXPECT_GT(results.size(), 0U);

        for (const auto& row : rows) {
            EXPECT_EQ(row.at("experiment_id").get<std::string>(), smoke_case.experiment_id);
            EXPECT_TRUE(row.at("correctness_pass").get<bool>());
        }

        for (const auto& result : results) {
            EXPECT_TRUE(result.at("experiment").get<std::string>().rfind(smoke_case.experiment_id, 0) == 0);
        }
    }
}
