#include "experiments/warp_level_coalescing_alignment_experiment.hpp"

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

constexpr const char* kExperimentId = "26_warp_level_coalescing_alignment";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kInputMultiplier = 17U;
constexpr uint32_t kInputOffset = 23U;
constexpr uint32_t kInputModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;
constexpr std::array<uint32_t, 6> kAlignmentOffsetsElements = {0U, 1U, 2U, 4U, 8U, 16U};

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
    uint32_t base_offset_elements = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t) * 2U);

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_variant_name(uint32_t offset_elements) {
    return "offset_" + std::to_string(offset_elements * sizeof(uint32_t)) + "b";
}

std::string make_case_name(uint32_t offset_elements, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(offset_elements) + "_elements_" +
           std::to_string(logical_count);
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputMultiplier) + kInputOffset) % kInputModulus;
}

uint32_t transform_value(uint32_t value, uint32_t logical_index) {
    return ((value + logical_index) * 1664525U) ^ 1013904223U;
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

std::vector<uint32_t> build_reference_values(const uint32_t* input_values, uint32_t logical_count,
                                             uint32_t offset_elements) {
    std::vector<uint32_t> reference(logical_count, 0U);
    for (uint32_t index = 0U; index < logical_count; ++index) {
        reference[index] = transform_value(input_values[offset_elements + index], index);
    }
    return reference;
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
    const uint64_t max_offset_elements = kAlignmentOffsetsElements.back();
    const uint64_t input_capacity_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    if (input_capacity_elements <= max_offset_elements) {
        return resolution;
    }

    const uint64_t input_limited_count = input_capacity_elements - max_offset_elements;
    const uint64_t output_limited_count = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count = std::min({input_limited_count, output_limited_count, dispatch_limited_count,
                                               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
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

VkDeviceSize compute_input_span_bytes(uint32_t logical_count) {
    const uint32_t max_offset_elements = kAlignmentOffsetsElements.back();
    return static_cast<VkDeviceSize>(logical_count + max_offset_elements) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_output_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize input_bytes, VkDeviceSize output_bytes,
                             BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), input_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create warp alignment input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create warp alignment output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "warp alignment input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "warp alignment output buffer",
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
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = input_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = output_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load warp alignment shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create warp alignment descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create warp alignment descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate warp alignment descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create warp alignment pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create warp alignment compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t logical_count,
                    uint32_t offset_elements) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        logical_count,
        offset_elements,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, uint32_t offset_elements, uint32_t logical_count, uint32_t group_count_x,
                       bool rounded_to_workgroup_multiple, VkDeviceSize input_span_bytes,
                       VkDeviceSize output_span_bytes, uint64_t logical_bytes_touched, bool correctness_pass,
                       bool dispatch_ok) {
    const uint32_t offset_bytes = offset_elements * sizeof(uint32_t);
    append_note(notes, "alignment_offset_elements=" + std::to_string(offset_elements));
    append_note(notes, "alignment_offset_bytes=" + std::to_string(offset_bytes));
    append_note(notes, "max_alignment_offset_elements=" + std::to_string(kAlignmentOffsetsElements.back()));
    append_note(notes,
                "max_alignment_offset_bytes=" + std::to_string(kAlignmentOffsetsElements.back() * sizeof(uint32_t)));
    append_note(notes, "access_pattern=contiguous_shifted");
    append_note(notes, "workgroup_size=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_input_elements=" + std::to_string(logical_count + kAlignmentOffsetsElements.back()));
    append_note(notes,
                "physical_input_span_bytes=" + std::to_string(static_cast<unsigned long long>(input_span_bytes)));
    append_note(notes, "output_span_bytes=" + std::to_string(static_cast<unsigned long long>(output_span_bytes)));
    append_note(notes,
                "logical_bytes_touched=" + std::to_string(static_cast<unsigned long long>(logical_bytes_touched)));
    append_note(notes, "bytes_per_logical_element=" + std::to_string(sizeof(uint32_t) * 2U));
    append_note(notes, "lane_group_proxy_width=32");
    append_note(notes, "alignment_baseline=" + std::string(offset_elements == 0U ? "true" : "false"));
    if (rounded_to_workgroup_multiple) {
        append_note(notes, "logical_count_rounded_to_workgroup_multiple=true");
    }
    append_note(notes, "validation_mode=transform_value_u32");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, uint32_t logical_count, uint32_t offset_elements,
              bool rounded_to_workgroup_multiple, WarpLevelCoalescingAlignmentExperimentOutput& output,
              bool verbose_progress) {
    const std::string variant_name = make_variant_name(offset_elements);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    const uint32_t input_elements = logical_count + kAlignmentOffsetsElements.back();
    const VkDeviceSize input_span_bytes = compute_input_span_bytes(logical_count);
    const VkDeviceSize output_span_bytes = compute_output_span_bytes(logical_count);
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers or illegal dispatch size for variant "
                  << variant_name << ".\n";
        return false;
    }

    const std::vector<uint32_t> reference_values = build_reference_values(input_values, logical_count, offset_elements);
    const uint64_t logical_bytes_touched = static_cast<uint64_t>(logical_count) * sizeof(uint32_t) * 2U;

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", offset_elements=" << offset_elements
                  << ", group_count_x=" << group_count_x << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, offset_elements);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, input_elements) &&
                             validate_output_values(output_values, reference_values);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, offset_elements);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, input_elements) &&
                             validate_output_values(output_values, reference_values);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, offset_elements, logical_count, group_count_x, rounded_to_workgroup_multiple,
                          input_span_bytes, output_span_bytes, logical_bytes_touched, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(logical_bytes_touched, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(make_case_name(offset_elements, logical_count), dispatch_samples));
    return true;
}

} // namespace

WarpLevelCoalescingAlignmentExperimentOutput
run_warp_level_coalescing_alignment_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                               const WarpLevelCoalescingAlignmentExperimentConfig& config) {
    WarpLevelCoalescingAlignmentExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "warp-level coalescing alignment experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "26_warp_level_coalescing_alignment.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for warp-level coalescing alignment experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for warp-level coalescing alignment experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = count_resolution.logical_count;
    const VkDeviceSize input_bytes = compute_input_span_bytes(logical_count);
    const VkDeviceSize output_bytes = compute_output_span_bytes(logical_count);

    BufferResources buffers{};
    if (!create_buffer_resources(context, input_bytes, output_bytes, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for input or output buffers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_input_values(input_values, logical_count + kAlignmentOffsetsElements.back());
    fill_output_values(output_values, logical_count);

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const uint32_t offset_elements : kAlignmentOffsetsElements) {
        if (!run_case(context, runner, buffers, pipeline_resources, logical_count, offset_elements,
                      count_resolution.rounded_to_workgroup_multiple, output, config.verbose_progress)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);
    destroy_buffer_resources(context, buffers);
    return output;
}
