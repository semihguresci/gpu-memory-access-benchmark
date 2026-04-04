#include "experiments/scatter_access_pattern_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "13_scatter_access_pattern";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr VkDeviceSize kBytesPerLogicalElement = static_cast<VkDeviceSize>(sizeof(uint32_t) * 3U);

enum class ScatterDistributionKind : std::uint8_t {
    kUniquePermutation,
    kRandomCollision,
    kClusteredHotset,
};

struct ScatterCaseDescriptor {
    ScatterDistributionKind kind;
};

constexpr std::array<ScatterCaseDescriptor, 3> kCaseDescriptors = {{
    {ScatterDistributionKind::kUniquePermutation},
    {ScatterDistributionKind::kRandomCollision},
    {ScatterDistributionKind::kClusteredHotset},
}};

struct ScatterCaseReference {
    std::vector<uint32_t> target_reference;
    std::vector<uint32_t> expected_counters;
    uint32_t active_target_count = 0U;
    uint32_t hot_target_count = 0U;
    uint32_t collision_factor = 1U;
    uint32_t max_expected_counter = 0U;
};

struct CaseBufferResources {
    BufferResource target_buffer{};
    BufferResource dst_buffer{};
    void* target_mapped_ptr = nullptr;
    void* dst_mapped_ptr = nullptr;
};

struct PipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_variant_name(ScatterDistributionKind kind, const ScatterAccessPatternExperimentConfig& config) {
    switch (kind) {
    case ScatterDistributionKind::kUniquePermutation:
        return "unique_permutation";
    case ScatterDistributionKind::kRandomCollision:
        return "random_collision_x" + std::to_string(config.collision_factor);
    case ScatterDistributionKind::kClusteredHotset:
        return "clustered_hotset_" + std::to_string(config.hot_window_size);
    }

    return "unknown";
}

std::string make_case_name(const std::string& variant_name, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + variant_name + "_elements_" + std::to_string(logical_count);
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t buffer_limited_count = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count_u64 = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    return static_cast<uint32_t>(effective_count_u64);
}

VkDeviceSize compute_physical_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_logical_bytes(uint32_t logical_count, uint32_t dispatch_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(dispatch_count) *
           kBytesPerLogicalElement;
}

double compute_effective_gbps(uint32_t logical_count, uint32_t dispatch_count, double dispatch_ms) {
    if (!std::isfinite(dispatch_ms) || dispatch_ms <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(compute_logical_bytes(logical_count, dispatch_count)) / (dispatch_ms * 1.0e6);
}

bool create_case_buffer_resources(VulkanContext& context, VkDeviceSize buffer_size,
                                  CaseBufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.target_buffer)) {
        std::cerr << "Failed to create scatter target buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create scatter destination counter buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.target_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.target_buffer, "scatter target buffer",
                           out_resources.target_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.target_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "scatter destination counter buffer",
                           out_resources.dst_mapped_ptr)) {
        if (out_resources.target_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.target_buffer.memory);
            out_resources.target_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.target_buffer);
        return false;
    }

    return true;
}

void destroy_case_buffer_resources(VulkanContext& context, CaseBufferResources& resources) {
    if (resources.dst_mapped_ptr != nullptr && resources.dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.dst_buffer.memory);
        resources.dst_mapped_ptr = nullptr;
    }

    if (resources.target_mapped_ptr != nullptr && resources.target_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.target_buffer.memory);
        resources.target_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.dst_buffer);
    destroy_buffer_resource(context.device(), resources.target_buffer);
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load scatter access pattern shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create scatter access pattern descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create scatter access pattern descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate scatter access pattern descriptor set.\n";
        return false;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(uint32_t) * 2U)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create scatter access pattern pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create scatter access pattern compute pipeline.\n";
        return false;
    }

    return true;
}

void destroy_pipeline_resources(VulkanContext& context, PipelineResources& resources) {
    if (resources.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), resources.pipeline, nullptr);
        resources.pipeline = VK_NULL_HANDLE;
    }

    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }

    if (resources.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device(), resources.descriptor_pool, nullptr);
        resources.descriptor_pool = VK_NULL_HANDLE;
    }

    if (resources.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device(), resources.descriptor_set_layout, nullptr);
        resources.descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (resources.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), resources.shader_module, nullptr);
        resources.shader_module = VK_NULL_HANDLE;
    }

    resources.descriptor_set = VK_NULL_HANDLE;
}

void update_case_descriptor_set(VulkanContext& context, const PipelineResources& resources,
                                const CaseBufferResources& buffers) {
    const VkDescriptorBufferInfo target_info{
        buffers.target_buffer.buffer,
        0U,
        buffers.target_buffer.size,
    };
    const VkDescriptorBufferInfo dst_info{
        buffers.dst_buffer.buffer,
        0U,
        buffers.dst_buffer.size,
    };

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), resources.descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = target_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = dst_info,
                                                          },
                                                      });
}

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t logical_count,
                    uint32_t target_capacity) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::array<uint32_t, 2> push_constants{logical_count, target_capacity};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), push_constants.data());
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

std::vector<uint32_t> make_shuffled_indices(uint32_t logical_count, std::mt19937& engine) {
    std::vector<uint32_t> indices(logical_count);
    std::iota(indices.begin(), indices.end(), 0U);
    std::shuffle(indices.begin(), indices.end(), engine);
    return indices;
}

ScatterCaseReference make_case_reference(ScatterDistributionKind kind, uint32_t logical_count,
                                         const ScatterAccessPatternExperimentConfig& config) {
    ScatterCaseReference reference{};
    reference.target_reference.reserve(logical_count);

    std::mt19937 engine(config.pattern_seed);
    switch (kind) {
    case ScatterDistributionKind::kUniquePermutation:
        reference.target_reference = make_shuffled_indices(logical_count, engine);
        reference.collision_factor = 1U;
        break;

    case ScatterDistributionKind::kRandomCollision: {
        const uint32_t collision_factor = std::max(1U, config.collision_factor);
        const uint32_t active_target_count =
            std::max(1U, static_cast<uint32_t>((static_cast<uint64_t>(logical_count) + collision_factor - 1U) /
                                               collision_factor));

        std::vector<uint32_t> active_targets = make_shuffled_indices(logical_count, engine);
        active_targets.resize(active_target_count);

        reference.target_reference.resize(logical_count);
        for (uint32_t index = 0U; index < logical_count; ++index) {
            reference.target_reference[index] = active_targets[index % active_target_count];
        }

        std::shuffle(reference.target_reference.begin(), reference.target_reference.end(), engine);
        reference.collision_factor = collision_factor;
        break;
    }

    case ScatterDistributionKind::kClusteredHotset: {
        const uint32_t hot_window_size = std::min(std::max(1U, config.hot_window_size), logical_count);
        const uint32_t chunk_size = std::max(kWorkgroupSize, hot_window_size);

        std::vector<uint32_t> window_starts;
        for (uint32_t window_start = 0U; window_start < logical_count; window_start += hot_window_size) {
            window_starts.push_back(window_start);
        }
        std::shuffle(window_starts.begin(), window_starts.end(), engine);

        for (uint32_t output_index = 0U, chunk_index = 0U; output_index < logical_count; ++chunk_index) {
            const uint32_t window_start = window_starts[chunk_index % window_starts.size()];
            const uint32_t window_length = std::min(hot_window_size, logical_count - window_start);
            const uint32_t chunk_length = std::min(chunk_size, logical_count - output_index);

            std::vector<uint32_t> chunk_targets;
            chunk_targets.reserve(chunk_length);

            // Keep adjacent invocations focused on a small destination window to create localized contention.
            for (uint32_t local_index = 0U; local_index < chunk_length; ++local_index) {
                chunk_targets.push_back(window_start + (local_index % window_length));
            }

            std::shuffle(chunk_targets.begin(), chunk_targets.end(), engine);
            reference.target_reference.insert(reference.target_reference.end(), chunk_targets.begin(),
                                              chunk_targets.end());
            output_index += chunk_length;
        }

        reference.hot_target_count = hot_window_size;
        reference.collision_factor =
            std::max(1U, (std::min(chunk_size, logical_count) + hot_window_size - 1U) / hot_window_size);
        break;
    }
    }

    reference.expected_counters.assign(logical_count, 0U);
    for (const uint32_t target_index : reference.target_reference) {
        if (target_index >= logical_count) {
            reference.target_reference.clear();
            reference.expected_counters.clear();
            reference.active_target_count = 0U;
            reference.hot_target_count = 0U;
            reference.collision_factor = 0U;
            reference.max_expected_counter = 0U;
            return reference;
        }
        ++reference.expected_counters[target_index];
    }

    reference.active_target_count = static_cast<uint32_t>(
        std::count_if(reference.expected_counters.begin(), reference.expected_counters.end(), [](uint32_t count) {
            return count != 0U;
        }));
    if (!reference.expected_counters.empty()) {
        reference.max_expected_counter = *std::ranges::max_element(reference.expected_counters);
    }

    return reference;
}

void fill_zero_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, 0U);
}

bool validate_case_values(const uint32_t* target_values, const uint32_t* dst_values,
                          const ScatterCaseReference& case_reference, uint32_t element_count) {
    if (case_reference.target_reference.size() != element_count ||
        case_reference.expected_counters.size() != element_count) {
        return false;
    }

    for (uint32_t index = 0U; index < element_count; ++index) {
        if (target_values[index] != case_reference.target_reference[index]) {
            return false;
        }

        if (dst_values[index] != case_reference.expected_counters[index]) {
            return false;
        }
    }

    return true;
}

void record_case_run(std::string& notes, const std::string& variant_name,
                     const ScatterAccessPatternExperimentConfig& config, const ScatterCaseReference& case_reference,
                     uint32_t logical_count, VkDeviceSize physical_span_bytes, uint32_t group_count_x,
                     bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "distribution=" + variant_name);
    append_note(notes, "seed=" + std::to_string(config.pattern_seed));
    append_note(notes, "collision_factor=" + std::to_string(case_reference.collision_factor));
    append_note(notes, "hot_window_size=" + std::to_string(config.hot_window_size));
    append_note(notes, "hot_target_count=" + std::to_string(case_reference.hot_target_count));
    append_note(notes, "active_target_count=" + std::to_string(case_reference.active_target_count));
    append_note(notes, "max_expected_counter=" + std::to_string(case_reference.max_expected_counter));
    append_note(notes, "target_capacity=" + std::to_string(logical_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_span_bytes=" + std::to_string(static_cast<unsigned long long>(physical_span_bytes)));
    append_note(notes, "allocated_span_bytes=" + std::to_string(static_cast<unsigned long long>(physical_span_bytes)));
    append_note(notes, "bytes_per_logical_element=" +
                           std::to_string(static_cast<unsigned long long>(kBytesPerLogicalElement)));
    append_note(notes, "validation_mode=exact_uint32_histogram");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const ScatterAccessPatternExperimentConfig& config, ScatterDistributionKind kind, uint32_t logical_count,
              VkDeviceSize physical_span_bytes, ScatterAccessPatternExperimentOutput& output) {
    const std::string variant_name = make_variant_name(kind, config);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for variant " << variant_name
                  << ".\n";
        return false;
    }

    const ScatterCaseReference case_reference = make_case_reference(kind, logical_count, config);
    if (case_reference.target_reference.size() != logical_count ||
        case_reference.expected_counters.size() != logical_count || case_reference.collision_factor == 0U) {
        std::cerr << "[" << kExperimentId << "] Failed to build scatter reference for variant " << variant_name
                  << ".\n";
        return false;
    }

    CaseBufferResources buffers{};
    if (!create_case_buffer_resources(context, physical_span_bytes, buffers)) {
        destroy_case_buffer_resources(context, buffers);
        return false;
    }

    update_case_descriptor_set(context, pipeline_resources, buffers);

    auto* target_values = static_cast<uint32_t*>(buffers.target_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (target_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for variant " << variant_name << ".\n";
        destroy_case_buffer_resources(context, buffers);
        return false;
    }

    std::ranges::copy(case_reference.target_reference, target_values);

    const std::size_t timed_iterations = static_cast<std::size_t>(std::max(0, runner.timed_iterations()));
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(timed_iterations);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", physical_span_bytes=" << physical_span_bytes
                  << ", group_count_x=" << group_count_x << ", seed=" << config.pattern_seed
                  << ", collision_factor=" << case_reference.collision_factor
                  << ", hot_window_size=" << config.hot_window_size
                  << ", hot_target_count=" << case_reference.hot_target_count
                  << ", active_target_count=" << case_reference.active_target_count
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_zero_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_case_values(target_values, dst_values, case_reference, logical_count);
        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_elements=" << logical_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_zero_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_case_values(target_values, dst_values, case_reference, logical_count);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_run(notes, variant_name, config, case_reference, logical_count, physical_span_bytes, group_count_x,
                        correctness_pass, dispatch_ok);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(logical_count, kDispatchCount, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(variant_name, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    destroy_case_buffer_resources(context, buffers);
    return true;
}

} // namespace

ScatterAccessPatternExperimentOutput
run_scatter_access_pattern_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const ScatterAccessPatternExperimentConfig& config) {
    ScatterAccessPatternExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "scatter access pattern experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.collision_factor == 0U || config.hot_window_size == 0U) {
        std::cerr << "scatter access pattern experiment requires non-zero collision_factor and hot_window_size.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "13_scatter_access_pattern.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for scatter access pattern experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for scatter access pattern experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        const VkDeviceSize physical_span_bytes = compute_physical_span_bytes(logical_count);
        std::cout << "[" << kExperimentId << "] Shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_elements=" << logical_count
                  << ", physical_span_bytes_per_buffer=" << physical_span_bytes
                  << ", scratch_size_bytes=" << config.max_buffer_bytes << ", seed=" << config.pattern_seed
                  << ", collision_factor=" << config.collision_factor << ", hot_window_size=" << config.hot_window_size
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        output.all_points_correct = false;
        return output;
    }

    const VkDeviceSize physical_span_bytes = compute_physical_span_bytes(logical_count);
    for (const ScatterCaseDescriptor& case_descriptor : kCaseDescriptors) {
        if (!run_case(context, runner, pipeline_resources, config, case_descriptor.kind, logical_count,
                      physical_span_bytes, output)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
