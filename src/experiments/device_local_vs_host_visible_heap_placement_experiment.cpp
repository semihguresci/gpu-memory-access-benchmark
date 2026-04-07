#include "experiments/device_local_vs_host_visible_heap_placement_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "28_device_local_vs_host_visible_heap_placement";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;

enum class VariantKind : uint32_t {
    HostVisibleDirect,
    DeviceLocalStaged,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::HostVisibleDirect, "host_visible_direct"},
    {VariantKind::DeviceLocalStaged, "device_local_staged"},
}};

struct BufferResources {
    BufferResource host_src_buffer{};
    BufferResource host_dst_buffer{};
    BufferResource staging_src_buffer{};
    BufferResource staging_dst_buffer{};
    BufferResource device_src_buffer{};
    BufferResource device_dst_buffer{};
    void* host_src_mapped_ptr = nullptr;
    void* host_dst_mapped_ptr = nullptr;
    void* staging_src_mapped_ptr = nullptr;
    void* staging_dst_mapped_ptr = nullptr;
};

struct PipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet host_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSet device_descriptor_set = VK_NULL_HANDLE;
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

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus;
}

uint32_t transform_value(uint32_t input_value, uint32_t index) {
    return ((input_value ^ (index * 1664525U)) + 1013904223U);
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

std::vector<uint32_t> build_reference_values(const uint32_t* input_values, uint32_t logical_count) {
    std::vector<uint32_t> reference(logical_count, 0U);
    for (uint32_t index = 0U; index < logical_count; ++index) {
        reference[index] = transform_value(input_values[index], index);
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

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t max_from_buffer = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t max_from_dispatch = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count =
        std::min({max_from_buffer, max_from_dispatch, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_count < kWorkgroupSize) {
        return 0U;
    }

    const uint64_t rounded = effective_count - (effective_count % kWorkgroupSize);
    return rounded >= kWorkgroupSize ? static_cast<uint32_t>(rounded) : 0U;
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.host_src_buffer)) {
        std::cerr << "Failed to create host-visible source buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.host_dst_buffer)) {
        std::cerr << "Failed to create host-visible destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.host_src_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_src_buffer)) {
        std::cerr << "Failed to create staging source buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.host_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_src_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging_dst_buffer)) {
        std::cerr << "Failed to create staging destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.staging_src_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_src_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.device_src_buffer)) {
        std::cerr << "Failed to create device-local source buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.staging_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.staging_src_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_src_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.device_dst_buffer)) {
        std::cerr << "Failed to create device-local destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.device_src_buffer);
        destroy_buffer_resource(context.device(), out_resources.staging_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.staging_src_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.host_src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.host_src_buffer, "host-visible source buffer",
                           out_resources.host_src_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.host_dst_buffer, "host-visible destination buffer",
                           out_resources.host_dst_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.staging_src_buffer, "staging source buffer",
                           out_resources.staging_src_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.staging_dst_buffer, "staging destination buffer",
                           out_resources.staging_dst_mapped_ptr)) {
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.staging_dst_mapped_ptr != nullptr && resources.staging_dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_dst_buffer.memory);
        resources.staging_dst_mapped_ptr = nullptr;
    }
    if (resources.staging_src_mapped_ptr != nullptr && resources.staging_src_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_src_buffer.memory);
        resources.staging_src_mapped_ptr = nullptr;
    }
    if (resources.host_dst_mapped_ptr != nullptr && resources.host_dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.host_dst_buffer.memory);
        resources.host_dst_mapped_ptr = nullptr;
    }
    if (resources.host_src_mapped_ptr != nullptr && resources.host_src_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.host_src_buffer.memory);
        resources.host_src_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.device_dst_buffer);
    destroy_buffer_resource(context.device(), resources.device_src_buffer);
    destroy_buffer_resource(context.device(), resources.staging_dst_buffer);
    destroy_buffer_resource(context.device(), resources.staging_src_buffer);
    destroy_buffer_resource(context.device(), resources.host_dst_buffer);
    destroy_buffer_resource(context.device(), resources.host_src_buffer);
}

void update_descriptor_set(VulkanContext& context, VkDescriptorSet descriptor_set, const BufferResource& src_buffer,
                           const BufferResource& dst_buffer) {
    const VkDescriptorBufferInfo src_info{src_buffer.buffer, 0U, src_buffer.size};
    const VkDescriptorBufferInfo dst_info{dst_buffer.buffer, 0U, dst_buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, src_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dst_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load heap placement shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create heap placement descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 2U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create heap placement descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.host_descriptor_set) ||
        !VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.device_descriptor_set)) {
        std::cerr << "Failed to allocate heap placement descriptor sets.\n";
        return false;
    }

    update_descriptor_set(context, out_resources.host_descriptor_set, buffers.host_src_buffer, buffers.host_dst_buffer);
    update_descriptor_set(context, out_resources.device_descriptor_set, buffers.device_src_buffer,
                          buffers.device_dst_buffer);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create heap placement pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create heap placement compute pipeline.\n";
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
    resources.host_descriptor_set = VK_NULL_HANDLE;
    resources.device_descriptor_set = VK_NULL_HANDLE;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, VkDescriptorSet descriptor_set,
                    uint32_t logical_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{logical_count};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

double run_upload_copy(VulkanContext& context, const BufferResource& src_buffer, const BufferResource& dst_buffer) {
    const VkBufferCopy region{0U, 0U, dst_buffer.size};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdCopyBuffer(command_buffer, src_buffer.buffer, dst_buffer.buffer, 1U, &region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, dst_buffer.buffer,
                                                                                dst_buffer.size);
    });
}

double run_readback_copy(VulkanContext& context, const BufferResource& src_buffer, const BufferResource& dst_buffer) {
    const VkBufferCopy region{0U, 0U, src_buffer.size};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, src_buffer.buffer,
                                                                          src_buffer.size);
        vkCmdCopyBuffer(command_buffer, src_buffer.buffer, dst_buffer.buffer, 1U, &region);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t logical_count,
                       std::size_t scratch_size_bytes, double upload_ms, double readback_ms, bool correctness_pass,
                       bool dispatch_ok) {
    append_note(notes, std::string("placement=") + descriptor.variant_name);
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    append_note(notes,
                "resident_buffer_count=" + std::to_string(descriptor.kind == VariantKind::HostVisibleDirect ? 2U : 4U));
    append_note(notes, "upload_ms=" + std::to_string(upload_ms));
    append_note(notes, "readback_ms=" + std::to_string(readback_ms));
    append_note(notes, "payload_bytes_per_element=8");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, const VariantDescriptor& descriptor, uint32_t logical_count,
              const std::vector<uint32_t>& reference_values, std::size_t scratch_size_bytes,
              DeviceLocalVsHostVisibleHeapPlacementExperimentOutput& output, bool verbose_progress) {
    auto* host_src_values = static_cast<uint32_t*>(buffers.host_src_mapped_ptr);
    auto* host_dst_values = static_cast<uint32_t*>(buffers.host_dst_mapped_ptr);
    auto* staging_src_values = static_cast<uint32_t*>(buffers.staging_src_mapped_ptr);
    auto* staging_dst_values = static_cast<uint32_t*>(buffers.staging_dst_mapped_ptr);
    if (host_src_values == nullptr || host_dst_values == nullptr || staging_src_values == nullptr ||
        staging_dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffer pointer for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));
    const uint64_t payload_bytes = static_cast<uint64_t>(logical_count) * sizeof(uint32_t) * 2U;

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        if (descriptor.kind == VariantKind::HostVisibleDirect) {
            fill_input_values(host_src_values, logical_count);
            fill_output_values(host_dst_values, logical_count);
            const double dispatch_ms =
                run_dispatch(context, pipeline_resources, pipeline_resources.host_descriptor_set, logical_count);
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool data_ok = dispatch_ok && validate_input_values(host_src_values, logical_count) &&
                                 validate_output_values(host_dst_values, reference_values);
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                          << " variant=" << descriptor.variant_name << ", dispatch_ms=" << dispatch_ms
                          << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
            }
        } else {
            fill_input_values(staging_src_values, logical_count);
            fill_output_values(staging_dst_values, logical_count);
            const double upload_ms = run_upload_copy(context, buffers.staging_src_buffer, buffers.device_src_buffer);
            const double dispatch_ms =
                run_dispatch(context, pipeline_resources, pipeline_resources.device_descriptor_set, logical_count);
            const double readback_ms =
                run_readback_copy(context, buffers.device_dst_buffer, buffers.staging_dst_buffer);
            const bool dispatch_ok =
                std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
            const bool data_ok = dispatch_ok && validate_input_values(staging_src_values, logical_count) &&
                                 validate_output_values(staging_dst_values, reference_values);
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                          << " variant=" << descriptor.variant_name << ", upload_ms=" << upload_ms
                          << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                          << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
            }
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        double upload_ms = 0.0;
        double dispatch_ms = std::numeric_limits<double>::quiet_NaN();
        double readback_ms = 0.0;
        bool correctness_pass = false;

        if (descriptor.kind == VariantKind::HostVisibleDirect) {
            fill_input_values(host_src_values, logical_count);
            fill_output_values(host_dst_values, logical_count);
            dispatch_ms =
                run_dispatch(context, pipeline_resources, pipeline_resources.host_descriptor_set, logical_count);
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            correctness_pass = dispatch_ok && validate_input_values(host_src_values, logical_count) &&
                               validate_output_values(host_dst_values, reference_values);
        } else {
            fill_input_values(staging_src_values, logical_count);
            fill_output_values(staging_dst_values, logical_count);
            upload_ms = run_upload_copy(context, buffers.staging_src_buffer, buffers.device_src_buffer);
            dispatch_ms =
                run_dispatch(context, pipeline_resources, pipeline_resources.device_descriptor_set, logical_count);
            readback_ms = run_readback_copy(context, buffers.device_dst_buffer, buffers.staging_dst_buffer);
            const bool dispatch_ok =
                std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
            correctness_pass = dispatch_ok && validate_input_values(staging_src_values, logical_count) &&
                               validate_output_values(staging_dst_values, reference_values);
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, logical_count, scratch_size_bytes, upload_ms, readback_ms,
                          correctness_pass, dispatch_ok);

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

DeviceLocalVsHostVisibleHeapPlacementExperimentOutput run_device_local_vs_host_visible_heap_placement_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const DeviceLocalVsHostVisibleHeapPlacementExperimentConfig& config) {
    DeviceLocalVsHostVisibleHeapPlacementExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "device-local vs host-visible heap placement experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "28_heap_placement_copy.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for heap placement experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for heap placement experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_span_bytes(logical_count), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* host_src_values = static_cast<uint32_t*>(buffers.host_src_mapped_ptr);
    if (host_src_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped host source buffer pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_input_values(host_src_values, logical_count);
    const std::vector<uint32_t> reference_values = build_reference_values(host_src_values, logical_count);

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const VariantDescriptor& descriptor : kVariantDescriptors) {
        if (!run_case(context, runner, buffers, pipeline_resources, descriptor, logical_count, reference_values,
                      config.scratch_size_bytes, output, config.verbose_progress)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);
    destroy_buffer_resources(context, buffers);
    return output;
}
