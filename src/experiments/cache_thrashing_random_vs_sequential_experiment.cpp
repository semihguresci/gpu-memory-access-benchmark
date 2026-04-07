#include "experiments/cache_thrashing_random_vs_sequential_experiment.hpp"

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

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "27_cache_thrashing_random_vs_sequential";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;
constexpr uint32_t kBlockShuffleWidth = 32U;
constexpr uint32_t kPatternSeed = 0x13579BDFU;

enum class AccessPatternKind : uint32_t {
    Sequential = 0U,
    BlockShuffled = 1U,
    Random = 2U,
};

struct PatternDescriptor {
    AccessPatternKind kind;
    const char* variant_name;
};

constexpr std::array<PatternDescriptor, 3> kPatternDescriptors = {{
    {AccessPatternKind::Sequential, "sequential"},
    {AccessPatternKind::BlockShuffled, "block_shuffled"},
    {AccessPatternKind::Random, "random"},
}};

struct CountResolution {
    uint32_t logical_count = 0U;
    bool rounded_to_workgroup_multiple = false;
};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource indices_buffer{};
    BufferResource output_buffer{};
    void* input_mapped_ptr = nullptr;
    void* indices_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
};

struct PipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t logical_count = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_case_name(AccessPatternKind kind, uint32_t logical_count) {
    for (const auto& descriptor : kPatternDescriptors) {
        if (descriptor.kind == kind) {
            return std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" +
                   std::to_string(logical_count);
        }
    }

    return std::string(kExperimentId) + "_unknown_elements_" + std::to_string(logical_count);
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus;
}

uint32_t transform_value(uint32_t input_value, uint32_t source_index) {
    return ((input_value ^ (source_index * 1664525U)) + 1013904223U);
}

void fill_input_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = input_pattern_value(index);
    }
}

void fill_output_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, kOutputSentinelValue);
}

bool validate_input_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != input_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> build_indices(AccessPatternKind kind, uint32_t logical_count) {
    std::vector<uint32_t> indices(logical_count, 0U);
    std::iota(indices.begin(), indices.end(), 0U);

    if (kind == AccessPatternKind::Sequential || logical_count == 0U) {
        return indices;
    }

    if (kind == AccessPatternKind::BlockShuffled) {
        std::vector<uint32_t> block_ids((logical_count + kBlockShuffleWidth - 1U) / kBlockShuffleWidth, 0U);
        std::iota(block_ids.begin(), block_ids.end(), 0U);

        std::mt19937 engine(kPatternSeed);
        std::shuffle(block_ids.begin(), block_ids.end(), engine);

        std::vector<uint32_t> shuffled;
        shuffled.reserve(indices.size());
        for (const uint32_t block_id : block_ids) {
            const uint32_t block_start = block_id * kBlockShuffleWidth;
            const uint32_t block_end = std::min(logical_count, block_start + kBlockShuffleWidth);
            for (uint32_t index = block_start; index < block_end; ++index) {
                shuffled.push_back(index);
            }
        }
        return shuffled;
    }

    std::mt19937 engine(kPatternSeed);
    std::shuffle(indices.begin(), indices.end(), engine);
    return indices;
}

std::vector<uint32_t> build_reference_values(const uint32_t* input_values, const std::vector<uint32_t>& indices) {
    std::vector<uint32_t> reference(indices.size(), 0U);
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const uint32_t source_index = indices[index];
        reference[index] = transform_value(input_values[source_index], source_index);
    }
    return reference;
}

bool validate_index_values(const uint32_t* values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (values[index] != reference_values[index]) {
            return false;
        }
    }
    return true;
}

bool validate_output_values(const uint32_t* values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (values[index] != reference_values[index]) {
            return false;
        }
    }
    return true;
}

CountResolution determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    CountResolution resolution{};
    const uint64_t buffer_limited_count = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_count < kWorkgroupSize) {
        return resolution;
    }

    const uint64_t rounded_count = effective_count - (effective_count % kWorkgroupSize);
    if (rounded_count == 0U) {
        return resolution;
    }

    resolution.logical_count = static_cast<uint32_t>(rounded_count);
    resolution.rounded_to_workgroup_multiple = rounded_count != effective_count;
    return resolution;
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create cache thrashing input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.indices_buffer)) {
        std::cerr << "Failed to create cache thrashing indices buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create cache thrashing output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.indices_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "cache thrashing input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.indices_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.indices_buffer, "cache thrashing indices buffer",
                           out_resources.indices_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.indices_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "cache thrashing output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.indices_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.indices_buffer.memory);
            out_resources.indices_mapped_ptr = nullptr;
        }
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.indices_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.indices_mapped_ptr != nullptr && resources.indices_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.indices_buffer.memory);
        resources.indices_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.indices_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo indices_info{buffers.indices_buffer.buffer, 0U, buffers.indices_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = input_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = indices_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = output_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load cache thrashing shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create cache thrashing descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create cache thrashing descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate cache thrashing descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create cache thrashing pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create cache thrashing compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t logical_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{logical_count};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

const char* pattern_name(AccessPatternKind kind) {
    for (const auto& descriptor : kPatternDescriptors) {
        if (descriptor.kind == kind) {
            return descriptor.variant_name;
        }
    }
    return "unknown";
}

void record_case_notes(std::string& notes, AccessPatternKind kind, uint32_t logical_count, uint32_t group_count_x,
                       bool rounded_to_workgroup_multiple, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("pattern=") + pattern_name(kind));
    append_note(notes, "seed=" + std::to_string(kPatternSeed));
    append_note(notes, "block_shuffle_width=" + std::to_string(kBlockShuffleWidth));
    append_note(notes, "working_set_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_span_bytes(logical_count))));
    append_note(notes, "workgroup_size=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "payload_bytes_per_element=12");
    if (rounded_to_workgroup_multiple) {
        append_note(notes, "logical_count_rounded_to_workgroup_multiple=true");
    }
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, AccessPatternKind kind, uint32_t logical_count,
              bool rounded_to_workgroup_multiple, CacheThrashingRandomVsSequentialExperimentOutput& output,
              bool verbose_progress) {
    const std::vector<uint32_t> indices = build_indices(kind, logical_count);
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* index_values = static_cast<uint32_t*>(buffers.indices_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (input_values == nullptr || index_values == nullptr || output_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers or illegal dispatch size for pattern "
                  << pattern_name(kind) << ".\n";
        return false;
    }

    std::copy(indices.begin(), indices.end(), index_values);
    const std::vector<uint32_t> reference_values = build_reference_values(input_values, indices);
    const uint64_t payload_bytes = static_cast<uint64_t>(logical_count) * sizeof(uint32_t) * 3U;

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, logical_count) &&
                             validate_index_values(index_values, indices) &&
                             validate_output_values(output_values, reference_values);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << pattern_name(kind) << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << pattern_name(kind)
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, logical_count) &&
                             validate_index_values(index_values, indices) &&
                             validate_output_values(output_values, reference_values);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, kind, logical_count, group_count_x, rounded_to_workgroup_multiple, correctness_pass,
                          dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = pattern_name(kind),
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(make_case_name(kind, logical_count), dispatch_samples));
    return true;
}

} // namespace

CacheThrashingRandomVsSequentialExperimentOutput
run_cache_thrashing_random_vs_sequential_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const CacheThrashingRandomVsSequentialExperimentConfig& config) {
    CacheThrashingRandomVsSequentialExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "cache thrashing experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "27_cache_thrashing_random_vs_sequential.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for cache thrashing experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for cache thrashing experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = count_resolution.logical_count;
    const VkDeviceSize span_bytes = compute_span_bytes(logical_count);

    BufferResources buffers{};
    if (!create_buffer_resources(context, span_bytes, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped input or output buffer pointers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_input_values(input_values, logical_count);
    fill_output_values(output_values, logical_count);

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const auto& descriptor : kPatternDescriptors) {
        if (!run_case(context, runner, buffers, pipeline_resources, descriptor.kind, logical_count,
                      count_resolution.rounded_to_workgroup_multiple, output, config.verbose_progress)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);
    destroy_buffer_resources(context, buffers);
    return output;
}
