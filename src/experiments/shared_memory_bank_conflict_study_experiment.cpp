#include "experiments/shared_memory_bank_conflict_study_experiment.hpp"

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
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "29_shared_memory_bank_conflict_study";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kInnerIterations = 16U;
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;
constexpr uint32_t kMaxStrideElements = 33U;

enum class VariantKind : uint32_t {
    Stride1 = 1U,
    Stride2 = 2U,
    Stride4 = 4U,
    Stride8 = 8U,
    Stride16 = 16U,
    Stride32 = 32U,
    PaddedFix = 33U,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    uint32_t stride_elements;
    bool padding_fix;
};

constexpr std::array<VariantDescriptor, 7> kVariantDescriptors = {{
    {VariantKind::Stride1, "stride_1", 1U, false},
    {VariantKind::Stride2, "stride_2", 2U, false},
    {VariantKind::Stride4, "stride_4", 4U, false},
    {VariantKind::Stride8, "stride_8", 8U, false},
    {VariantKind::Stride16, "stride_16", 16U, false},
    {VariantKind::Stride32, "stride_32", 32U, false},
    {VariantKind::PaddedFix, "padded_fix", 33U, true},
}};

struct CountResolution {
    uint32_t logical_count = 0U;
    bool rounded_to_workgroup_multiple = false;
};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource output_buffer{};
    void* input_mapped_ptr = nullptr;
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
    uint32_t stride_elements = 0U;
    uint32_t inner_iterations = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 3U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus;
}

void fill_input_values(uint32_t* values, uint32_t logical_count) {
    for (uint32_t index = 0U; index < logical_count; ++index) {
        values[index] = input_pattern_value(index);
    }
}

void fill_output_values(uint32_t* values, uint32_t logical_count) {
    std::fill_n(values, logical_count, kOutputSentinelValue);
}

bool validate_input_values(const uint32_t* values, uint32_t logical_count) {
    for (uint32_t index = 0U; index < logical_count; ++index) {
        if (values[index] != input_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> build_reference_values(const uint32_t* input_values, uint32_t logical_count,
                                             uint32_t stride_elements) {
    std::vector<uint32_t> reference(logical_count, 0U);
    std::vector<uint32_t> shared_values(static_cast<std::size_t>(kWorkgroupSize) * stride_elements, 0U);

    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    for (uint32_t group_index = 0U; group_index < group_count_x; ++group_index) {
        std::fill(shared_values.begin(), shared_values.end(), 0U);
        const uint32_t group_base = group_index * kWorkgroupSize;
        for (uint32_t local_index = 0U; local_index < kWorkgroupSize; ++local_index) {
            const uint32_t global_index = group_base + local_index;
            const uint32_t shared_index = local_index * stride_elements;
            shared_values[shared_index] = global_index < logical_count ? input_values[global_index] : 0U;
        }

        for (uint32_t local_index = 0U; local_index < kWorkgroupSize; ++local_index) {
            const uint32_t global_index = group_base + local_index;
            if (global_index >= logical_count) {
                continue;
            }

            uint32_t accumulator = input_values[global_index];
            for (uint32_t iteration = 0U; iteration < kInnerIterations; ++iteration) {
                const uint32_t read_local = (local_index + iteration) % kWorkgroupSize;
                const uint32_t read_index = read_local * stride_elements;
                accumulator = (accumulator * 1664525U) + shared_values[read_index] + 1013904223U + iteration;
            }
            reference[global_index] = accumulator;
        }
    }

    return reference;
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (output_values[index] != reference_values[index]) {
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
        std::cerr << "Failed to create shared bank conflict input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create shared bank conflict output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "shared bank conflict input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "shared bank conflict output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
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
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load shared bank conflict shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create shared bank conflict descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create shared bank conflict descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate shared bank conflict descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create shared bank conflict pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create shared bank conflict compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t logical_count,
                    uint32_t stride_elements) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{logical_count, stride_elements, kInnerIterations};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t logical_count,
                       uint32_t group_count_x, bool rounded_to_workgroup_multiple, bool correctness_pass,
                       bool dispatch_ok, std::size_t scratch_size_bytes) {
    append_note(notes, std::string("variant_kind=") + descriptor.variant_name);
    append_note(notes, "shared_stride_elements=" + std::to_string(descriptor.stride_elements));
    append_note(notes, "padding_fix=" + std::string(descriptor.padding_fix ? "true" : "false"));
    append_note(notes, "shared_span_elements=" + std::to_string(descriptor.stride_elements * kWorkgroupSize));
    append_note(notes, "shared_span_bytes=" + std::to_string(static_cast<unsigned long long>(
                                                  descriptor.stride_elements * kWorkgroupSize * sizeof(uint32_t))));
    append_note(notes, "inner_iterations=" + std::to_string(kInnerIterations));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "payload_bytes_per_element=8");
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
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
              const PipelineResources& pipeline_resources, const VariantDescriptor& descriptor, uint32_t logical_count,
              bool rounded_to_workgroup_multiple, SharedMemoryBankConflictStudyExperimentOutput& output,
              bool verbose_progress, std::size_t scratch_size_bytes) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (input_values == nullptr || output_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped pointers or invalid dispatch for variant=" << descriptor.variant_name << ".\n";
        return false;
    }

    const std::vector<uint32_t> reference_values =
        build_reference_values(input_values, logical_count, descriptor.stride_elements);
    const uint64_t payload_bytes = static_cast<uint64_t>(logical_count) * sizeof(uint32_t) * 2U;
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, descriptor.stride_elements);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, logical_count) &&
                             validate_output_values(output_values, reference_values);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, descriptor.stride_elements);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass = dispatch_ok && validate_input_values(input_values, logical_count) &&
                                      validate_output_values(output_values, reference_values);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, logical_count, group_count_x, rounded_to_workgroup_multiple,
                          correctness_pass, dispatch_ok, scratch_size_bytes);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
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

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" + std::to_string(logical_count),
        dispatch_samples));
    return true;
}

} // namespace

SharedMemoryBankConflictStudyExperimentOutput
run_shared_memory_bank_conflict_study_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                 const SharedMemoryBankConflictStudyExperimentConfig& config) {
    SharedMemoryBankConflictStudyExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "shared memory bank conflict study requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "29_shared_memory_bank_conflict_study.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for shared memory bank conflict study.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for shared memory bank conflict study.\n";
        output.all_points_correct = false;
        return output;
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_span_bytes(count_resolution.logical_count), buffers)) {
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

    fill_input_values(input_values, count_resolution.logical_count);
    fill_output_values(output_values, count_resolution.logical_count);

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const VariantDescriptor& descriptor : kVariantDescriptors) {
        if (!run_case(context, runner, buffers, pipeline_resources, descriptor, count_resolution.logical_count,
                      count_resolution.rounded_to_workgroup_multiple, output, config.verbose_progress,
                      config.scratch_size_bytes)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);
    destroy_buffer_resources(context, buffers);
    return output;
}
