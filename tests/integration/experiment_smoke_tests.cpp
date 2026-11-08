#include "support/filesystem_test_utils.hpp"
#include "vulkan_context.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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
constexpr std::array<SmokeCase, 24> kSmokeCases{{
    {.experiment_id = "01_dispatch_basics", .size_arg = "4M"},
    {.experiment_id = "03_memory_copy_baseline", .size_arg = "48"},
    {.experiment_id = "06_aos_vs_soa", .size_arg = "32M"},
    {.experiment_id = "11_coalesced_vs_strided", .size_arg = "32M"},
    {.experiment_id = "15_bandwidth_saturation_sweep", .size_arg = "48"},
    {.experiment_id = "16_shared_memory_tiling", .size_arg = "32M"},
    // Four MiB used to over-dispatch the smallest Experiment 17 tile on
    // devices whose maximum x dispatch dimension is 65,535 workgroups.
    {.experiment_id = "17_tile_size_sweep", .size_arg = "4M"},
    {.experiment_id = "18_register_pressure_proxy", .size_arg = "4M"},
    {.experiment_id = "20_barrier_synchronization_cost", .size_arg = "4M"},
    {.experiment_id = "22_prefix_sum_scan", .size_arg = "4M"},
    {.experiment_id = "24_stream_compaction", .size_arg = "32M"},
    {.experiment_id = "27_cache_thrashing_random_vs_sequential", .size_arg = "64M"},
    {.experiment_id = "28_device_local_vs_host_visible_heap_placement", .size_arg = "4M"},
    {.experiment_id = "32_subgroup_stream_compaction_variants", .size_arg = "4M"},
    {.experiment_id = "34_radix_sort_gpu", .size_arg = "128M"},
    {.experiment_id = "37_frame_to_frame_coherence", .size_arg = "4M"},
    {.experiment_id = "39_tiled_light_assignment", .size_arg = "4M"},
    {.experiment_id = "41_subgroup_operations_study", .size_arg = "32M"},
    {.experiment_id = "42_async_compute_overlap", .size_arg = "32M"},
    {.experiment_id = "44_gpu_driven_pipeline_blocks", .size_arg = "4M"},
    {.experiment_id = "45_cross_gpu_reproducibility", .size_arg = "32M"},
    {.experiment_id = "46_dependent_load_latency", .size_arg = "4M"},
    {.experiment_id = "47_block_reorder_factorial", .size_arg = "128M"},
    {.experiment_id = "48_buffer_vs_image_representation", .size_arg = "4M"},
}};

constexpr uint64_t kFourMiB = 4U * 1024U * 1024U;

std::optional<std::string_view> note_value(std::string_view notes, std::string_view key) {
    while (!notes.empty()) {
        const std::size_t delimiter = notes.find(';');
        const std::string_view token = notes.substr(0U, delimiter);
        const std::size_t separator = token.find('=');
        if (separator != std::string_view::npos && token.substr(0U, separator) == key) {
            return token.substr(separator + 1U);
        }
        if (delimiter == std::string_view::npos) {
            break;
        }
        notes.remove_prefix(delimiter + 1U);
    }
    return std::nullopt;
}

std::optional<uint64_t> integer_note_value(std::string_view notes, std::string_view key) {
    const auto text = note_value(notes, key);
    if (!text.has_value()) {
        return std::nullopt;
    }
    uint64_t value = 0U;
    const auto result = std::from_chars(text->data(), text->data() + text->size(), value);
    if (result.ec != std::errc{} || result.ptr != text->data() + text->size()) {
        return std::nullopt;
    }
    return value;
}

void expect_heap_placement_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto scratch = integer_note_value(notes, "scratch_size_bytes");
    const auto live_count = integer_note_value(notes, "live_buffer_count");
    const auto live_payload = integer_note_value(notes, "live_payload_bytes");
    const auto buffer_span = integer_note_value(notes, "buffer_span_bytes");
    ASSERT_TRUE(scratch.has_value());
    ASSERT_TRUE(live_count.has_value());
    ASSERT_TRUE(live_payload.has_value());
    ASSERT_TRUE(buffer_span.has_value());
    EXPECT_EQ(*scratch, kFourMiB);
    EXPECT_EQ(*live_count, 6U);
    EXPECT_LE(*live_payload, *scratch);
    EXPECT_EQ(*buffer_span, row.at("problem_size").get<uint64_t>() * sizeof(uint32_t));
    EXPECT_EQ(*live_payload, *buffer_span * *live_count);
}

void expect_scan_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto blocks = integer_note_value(notes, "block_count");
    const auto block_elements = integer_note_value(notes, "block_elements");
    const auto estimated = integer_note_value(notes, "estimated_global_total_bytes");
    ASSERT_TRUE(blocks.has_value());
    ASSERT_TRUE(block_elements.has_value());
    ASSERT_TRUE(estimated.has_value());
    ASSERT_GT(*blocks, 0U);
    ASSERT_GT(*block_elements, 0U);
    const uint64_t elements = row.at("problem_size").get<uint64_t>();
    EXPECT_EQ(*blocks, (elements + *block_elements - 1U) / *block_elements);
    // Four full element passes, three block-array operations, one scratch
    // write, and 256 per-invocation carry requests per nonfirst workgroup.
    const uint64_t expected_bytes = (4U * elements + 3U * *blocks + 1U + 256U * (*blocks - 1U)) * sizeof(uint32_t);
    EXPECT_EQ(*estimated, expected_bytes);
    const double gpu_ms = row.at("gpu_ms").get<double>();
    ASSERT_TRUE(std::isfinite(gpu_ms));
    ASSERT_GT(gpu_ms, 0.0);
    EXPECT_NEAR(row.at("gbps").get<double>() * gpu_ms * 1.0e6, static_cast<double>(expected_bytes), 1.0);
    EXPECT_EQ(note_value(notes, "gbps_basis"), "estimated_source_level_storage_buffer_bytes");
    EXPECT_EQ(note_value(notes, "traffic_model"), "source_storage_buffer_operations_not_physical_dram");
}

void expect_compaction_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto survivors = integer_note_value(notes, "valid_count");
    const auto blocks = integer_note_value(notes, "block_count");
    ASSERT_TRUE(survivors.has_value());
    ASSERT_TRUE(blocks.has_value());
    EXPECT_LE(*survivors, row.at("problem_size").get<uint64_t>());
    const double gpu_ms = row.at("gpu_ms").get<double>();
    const double gbps = row.at("gbps").get<double>();
    ASSERT_TRUE(std::isfinite(gpu_ms));
    ASSERT_GT(gpu_ms, 0.0);
    ASSERT_TRUE(std::isfinite(gbps));
    // Only survivors load source values. Reconstruct bytes from the exported
    // bandwidth/time to catch the former all-input read overcount at low ratios.
    const uint64_t expected_bytes = ((*survivors * 2U) + *blocks) * sizeof(uint32_t);
    EXPECT_NEAR(gbps * gpu_ms * 1.0e6, static_cast<double>(expected_bytes), 1.0);
    EXPECT_EQ(note_value(notes, "gbps_mode"), "shader_payload_bytes");
}

void expect_frame_coherence_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto scratch = integer_note_value(notes, "scratch_size_bytes");
    const auto resident_payload = integer_note_value(notes, "resident_buffer_payload_bytes");
    const auto source = integer_note_value(notes, "active_source_bytes");
    const auto output = integer_note_value(notes, "active_output_bytes");
    const auto frames = integer_note_value(notes, "frames_per_iteration");
    ASSERT_TRUE(scratch.has_value());
    ASSERT_TRUE(resident_payload.has_value());
    ASSERT_TRUE(source.has_value());
    ASSERT_TRUE(output.has_value());
    ASSERT_TRUE(frames.has_value());
    EXPECT_EQ(*scratch, kFourMiB);
    EXPECT_EQ(*frames, 8U);
    EXPECT_EQ(row.at("dispatch_count").get<uint64_t>(), *frames);
    EXPECT_EQ(*source, row.at("problem_size").get<uint64_t>() * sizeof(uint32_t));
    EXPECT_EQ(*output, *source * *frames);
    EXPECT_LE(*source * 9U, *resident_payload);
    EXPECT_LE(*resident_payload, *scratch);
    EXPECT_EQ(note_value(notes, "oracle"), "all_eight_frame_outputs_each_iteration");
    EXPECT_EQ(note_value(notes, "frame_outputs"), "disjoint");
}

void expect_block_reorder_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto scratch = integer_note_value(notes, "scratch_size_bytes");
    const auto live_count = integer_note_value(notes, "live_buffer_count");
    const auto live_payload = integer_note_value(notes, "live_payload_bytes");
    const auto live_allocation = integer_note_value(notes, "live_allocation_bytes");
    const auto workload_span = integer_note_value(notes, "workload_buffer_span_bytes");
    const auto live_perm_span = integer_note_value(notes, "live_block_perm_span_bytes");
    const auto eviction = integer_note_value(notes, "eviction_buffer_bytes");
    ASSERT_TRUE(scratch.has_value());
    ASSERT_TRUE(live_count.has_value());
    ASSERT_TRUE(live_payload.has_value());
    ASSERT_TRUE(live_allocation.has_value());
    ASSERT_TRUE(workload_span.has_value());
    ASSERT_TRUE(live_perm_span.has_value());
    ASSERT_TRUE(eviction.has_value());
    EXPECT_EQ(*scratch, 128U * 1024U * 1024U);
    EXPECT_EQ(*live_count, 6U);
    EXPECT_EQ(*eviction, 64U * 1024U * 1024U);
    EXPECT_EQ(*live_payload, (*workload_span * 4U) + *live_perm_span + *eviction);
    EXPECT_LE(*live_payload, *scratch);
    EXPECT_GE(*live_allocation, *live_payload);
    const auto block_width = integer_note_value(notes, "block_width");
    const auto subgroup_size = integer_note_value(notes, "subgroup_size");
    ASSERT_TRUE(block_width.has_value());
    ASSERT_TRUE(subgroup_size.has_value());
    ASSERT_GT(*subgroup_size, 0U);
    ASSERT_GT(*block_width, *subgroup_size);
    const uint64_t element_count = row.at("problem_size").get<uint64_t>();
    EXPECT_EQ(element_count % *block_width, 0U);
    const double gpu_ms = row.at("gpu_ms").get<double>();
    const double gbps = row.at("gbps").get<double>();
    ASSERT_TRUE(std::isfinite(gpu_ms));
    ASSERT_GT(gpu_ms, 0.0);
    ASSERT_TRUE(std::isfinite(gbps));
    // Both block-order modes load a permutation entry for every invocation.
    // Reconstruct exported request bytes to catch per-block deduplication.
    const uint64_t expected_bytes = element_count * 3U * sizeof(uint32_t);
    EXPECT_NEAR(gbps * gpu_ms * 1.0e6, static_cast<double>(expected_bytes), 1.0);
    EXPECT_EQ(note_value(notes, "gbps_mode"), "estimated_shader_request_bytes");
    EXPECT_EQ(note_value(notes, "oracle"), "reference_complement_before_every_dispatch");
}

void expect_dependent_load_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto active = integer_note_value(notes, "active_payload_bytes");
    const auto resident = integer_note_value(notes, "live_payload_bytes");
    const auto span = integer_note_value(notes, "resident_buffer_span_bytes");
    ASSERT_TRUE(active.has_value());
    ASSERT_TRUE(resident.has_value());
    ASSERT_TRUE(span.has_value());
    EXPECT_EQ(*active, row.at("problem_size").get<uint64_t>() * 4U * sizeof(uint32_t));
    EXPECT_EQ(*resident, *span * 4U);
    EXPECT_EQ(*resident, kFourMiB);
    EXPECT_LE(*active, *resident);
}

void expect_pipeline_traffic_export_contract(const nlohmann::json& row, const std::string& notes) {
    const auto visible = integer_note_value(notes, "visible_count");
    const auto estimated = integer_note_value(notes, "estimated_global_total_bytes");
    ASSERT_TRUE(visible.has_value());
    ASSERT_TRUE(estimated.has_value());
    const uint64_t count = row.at("problem_size").get<uint64_t>();
    const bool staged = row.at("variant").get<std::string>() == "staged_three_dispatch";
    EXPECT_EQ(*estimated, staged ? (32U * count) + (4U * *visible) : 16U * count);
    EXPECT_EQ(note_value(notes, "gbps_mode"), "logical_payload_bytes");
}

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

TEST(ExperimentSmokeTests, RejectsBlockReorderBudgetThatCannotFitEvictionBuffer) {
    VulkanContext context;
    if (!context.initialize(false) || !context.gpu_timestamps_supported()) {
        GTEST_SKIP() << "A Vulkan device with GPU timestamps is required for experiment smoke tests.";
    }
    context.shutdown();

    const std::filesystem::path binary_path = benchmark_binary_path();
    ASSERT_TRUE(std::filesystem::exists(binary_path));
    TestSupport::ScopedTempDirectory temp_directory("gpu_memory_layout_block_reorder_budget");
    for (const char* size_arg : {"1M", "64M"}) {
        SCOPED_TRACE(size_arg);
        const SmokeCase smoke_case{.experiment_id = "47_block_reorder_factorial", .size_arg = size_arg};
        const std::filesystem::path output_base = temp_directory.path() / size_arg;
        EXPECT_NE(run_benchmark_smoke_case(binary_path, smoke_case, output_base), 0);
        EXPECT_FALSE(std::filesystem::exists(output_base.string() + ".json"));
    }
}

TEST(ExperimentSmokeTests, RejectsScratchBudgetsTooSmallForReadOnlyProbes) {
    VulkanContext context;
    if (!context.initialize(false) || !context.gpu_timestamps_supported()) {
        GTEST_SKIP() << "A Vulkan device with timestamps is required.";
    }
    context.shutdown();
    TestSupport::ScopedTempDirectory directory("gpu_memory_layout_probe_budget");
    for (const char* experiment : {"03_memory_copy_baseline", "15_bandwidth_saturation_sweep"}) {
        for (const char* size_arg : {"12", "47"}) {
            SCOPED_TRACE(std::string(experiment) + ": " + size_arg);
            const std::filesystem::path output = directory.path() / (std::string(experiment) + size_arg);
            EXPECT_NE(run_benchmark_smoke_case(benchmark_binary_path(), {experiment, size_arg}, output), 0);
            EXPECT_FALSE(std::filesystem::exists(output.string() + ".json"));
        }
    }
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
            SCOPED_TRACE(std::string(smoke_case.experiment_id) + ": " + row.at("variant").get<std::string>());
            EXPECT_EQ(row.at("experiment_id").get<std::string>(), smoke_case.experiment_id);
            EXPECT_TRUE(row.at("correctness_pass").get<bool>());
            const std::string notes = row.at("notes").get<std::string>();
            const std::string_view experiment_id = smoke_case.experiment_id;
            if (experiment_id == "22_prefix_sum_scan") {
                expect_scan_export_contract(row, notes);
            } else if (experiment_id == "28_device_local_vs_host_visible_heap_placement") {
                expect_heap_placement_export_contract(row, notes);
            } else if (experiment_id == "32_subgroup_stream_compaction_variants") {
                expect_compaction_export_contract(row, notes);
            } else if (experiment_id == "37_frame_to_frame_coherence") {
                expect_frame_coherence_export_contract(row, notes);
            } else if (experiment_id == "47_block_reorder_factorial") {
                expect_block_reorder_export_contract(row, notes);
            } else if (experiment_id == "46_dependent_load_latency") {
                expect_dependent_load_export_contract(row, notes);
            } else if (experiment_id == "44_gpu_driven_pipeline_blocks") {
                expect_pipeline_traffic_export_contract(row, notes);
            }
        }

        for (const auto& result : results) {
            EXPECT_TRUE(result.at("experiment").get<std::string>().rfind(smoke_case.experiment_id, 0) == 0);
        }
    }
}
