#include "experiments/shared_memory_tiling_experiment.hpp"

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

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "16_shared_memory_tiling";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kMaxRadius = 16U;
constexpr uint32_t kCenterOffset = kMaxRadius;
constexpr uint32_t kSourcePatternMultiplier = 17U;
constexpr uint32_t kSourcePatternOffset = 23U;
constexpr uint32_t kSourcePatternModulus = 251U;
constexpr uint32_t kDestinationSentinelValue = 0xA5A5A5A5U;
constexpr uint32_t kSharedTileAllocatedElements = kWorkgroupSize + (2U * kMaxRadius);
constexpr VkDeviceSize kSharedBytesPerWorkgroup =
    static_cast<VkDeviceSize>(kSharedTileAllocatedElements) * static_cast<VkDeviceSize>(sizeof(uint32_t));
constexpr std::array<uint32_t, 4> kReuseRadii = {1U, 4U, 8U, 16U};

enum class ImplementationKind {
    DirectGlobal,
    SharedTiled,
};

struct ImplementationDescriptor {
    ImplementationKind kind;
    const char* implementation_name;
    const char* shader_filename;
};

constexpr std::array<ImplementationDescriptor, 2> kImplementationDescriptors = {{
    {ImplementationKind::DirectGlobal, "direct_global", "16_shared_memory_tiling_direct.comp.spv"},
    {ImplementationKind::SharedTiled, "shared_tiled", "16_shared_memory_tiling_tiled.comp.spv"},
}};

struct CountResolution {
    uint32_t logical_count = 0U;
    bool rounded_to_workgroup_multiple = false;
};

struct BufferResources {
    BufferResource src_buffer{};
    BufferResource dst_buffer{};
    void* src_mapped_ptr = nullptr;
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

struct PushConstants {
    uint32_t output_count = 0U;
    uint32_t radius = 0U;
    uint32_t center_offset = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 3U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_variant_name(ImplementationKind kind, uint32_t radius) {
    const char* implementation_name = "unknown";
    switch (kind) {
    case ImplementationKind::DirectGlobal:
        implementation_name = "direct_global";
        break;
    case ImplementationKind::SharedTiled:
        implementation_name = "shared_tiled";
        break;
    }

    return std::string(implementation_name) + "_r" + std::to_string(radius);
}

std::string make_case_name(ImplementationKind kind, uint32_t radius, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(kind, radius) + "_outputs_" +
           std::to_string(logical_count);
}

uint32_t source_pattern_value(uint32_t index) {
    return ((index * kSourcePatternMultiplier) + kSourcePatternOffset) % kSourcePatternModulus;
}

void fill_source_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_destination_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, kDestinationSentinelValue);
}

bool validate_source_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != source_pattern_value(index)) {
            return false;
        }
    }

    return true;
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (output_values[index] != reference_values[index]) {
            return false;
        }
    }

    return true;
}

std::vector<uint32_t> build_reference_values(const uint32_t* src_values, uint32_t logical_count, uint32_t radius) {
    std::vector<uint32_t> reference_values(logical_count, 0U);

    for (uint32_t logical_index = 0U; logical_index < logical_count; ++logical_index) {
        const uint32_t center_index = logical_index + kCenterOffset;
        uint32_t sum = 0U;
        for (uint32_t offset = 0U; offset < ((2U * radius) + 1U); ++offset) {
            sum += src_values[(center_index - radius) + offset];
        }
        reference_values[logical_index] = sum;
    }

    return reference_values;
}

CountResolution determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    CountResolution resolution{};
    const uint64_t source_capacity_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    if (source_capacity_elements <= (2U * kMaxRadius)) {
        return resolution;
    }

    const uint64_t source_limited_count = source_capacity_elements - (2U * kMaxRadius);
    const uint64_t output_limited_count = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count = std::min({source_limited_count, output_limited_count, dispatch_limited_count,
                                               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_count < kWorkgroupSize) {
        return resolution;
    }

    const uint64_t rounded_count = effective_count - (effective_count % kWorkgroupSize);
    resolution.logical_count = static_cast<uint32_t>(rounded_count);
    resolution.rounded_to_workgroup_multiple = (rounded_count != effective_count);
    return resolution;
}

uint32_t compute_source_padded_elements(uint32_t logical_count) {
    return logical_count + (2U * kMaxRadius);
}

VkDeviceSize compute_source_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(compute_source_padded_elements(logical_count)) *
           static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_output_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t compute_active_tile_span(uint32_t radius) {
    return kWorkgroupSize + (2U * radius);
}

VkDeviceSize compute_estimated_global_read_bytes(ImplementationKind kind, uint32_t logical_count, uint32_t radius,
                                                 uint32_t group_count_x) {
    switch (kind) {
    case ImplementationKind::DirectGlobal:
        return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>((2U * radius) + 1U) *
               static_cast<VkDeviceSize>(sizeof(uint32_t));
    case ImplementationKind::SharedTiled:
        return static_cast<VkDeviceSize>(group_count_x) * static_cast<VkDeviceSize>(compute_active_tile_span(radius)) *
               static_cast<VkDeviceSize>(sizeof(uint32_t));
    }

    return 0U;
}

VkDeviceSize compute_estimated_global_write_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

double compute_effective_gbps(VkDeviceSize estimated_read_bytes, VkDeviceSize estimated_write_bytes,
                              double dispatch_ms) {
    if (!std::isfinite(dispatch_ms) || dispatch_ms <= 0.0) {
        return 0.0;
    }

    const double total_bytes = static_cast<double>(estimated_read_bytes) + static_cast<double>(estimated_write_bytes);
    return total_bytes / (dispatch_ms * 1.0e6);
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize source_bytes, VkDeviceSize output_bytes,
                             BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), source_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.src_buffer)) {
        std::cerr << "Failed to create shared memory tiling source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create shared memory tiling destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "shared memory tiling source buffer",
                           out_resources.src_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "shared memory tiling destination buffer",
                           out_resources.dst_mapped_ptr)) {
        if (out_resources.src_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.src_buffer.memory);
            out_resources.src_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.dst_mapped_ptr != nullptr && resources.dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.dst_buffer.memory);
        resources.dst_mapped_ptr = nullptr;
    }

    if (resources.src_mapped_ptr != nullptr && resources.src_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.src_buffer.memory);
        resources.src_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.dst_buffer);
    destroy_buffer_resource(context.device(), resources.src_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo src_info{
        buffers.src_buffer.buffer,
        0U,
        buffers.src_buffer.size,
    };
    const VkDescriptorBufferInfo dst_info{
        buffers.dst_buffer.buffer,
        0U,
        buffers.dst_buffer.size,
    };

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = src_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = dst_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load shared memory tiling shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create shared memory tiling descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create shared memory tiling descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate shared memory tiling descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create shared memory tiling pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create shared memory tiling compute pipeline.\n";
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
                    uint32_t radius) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        logical_count,
        radius,
        kCenterOffset,
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

void record_case_notes(std::string& notes, ImplementationKind kind, uint32_t radius, uint32_t logical_count,
                       uint32_t group_count_x, bool rounded_to_workgroup_multiple, bool correctness_pass,
                       bool dispatch_ok, std::size_t scratch_size_bytes) {
    const uint32_t source_padded_elements = compute_source_padded_elements(logical_count);
    const uint32_t active_tile_span_elements = compute_active_tile_span(radius);
    const VkDeviceSize estimated_global_read_bytes =
        compute_estimated_global_read_bytes(kind, logical_count, radius, group_count_x);
    const VkDeviceSize estimated_global_write_bytes = compute_estimated_global_write_bytes(logical_count);
    const bool is_tiled = kind == ImplementationKind::SharedTiled;

    append_note(notes, std::string("implementation=") + (is_tiled ? "shared_tiled" : "direct_global"));
    append_note(notes, "reuse_radius=" + std::to_string(radius));
    append_note(notes, "center_offset=" + std::to_string(kCenterOffset));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "source_padded_elements=" + std::to_string(source_padded_elements));
    append_note(notes, "output_elements=" + std::to_string(logical_count));
    append_note(notes, "active_tile_span_elements=" + std::to_string(active_tile_span_elements));
    append_note(notes, "shared_bytes_per_workgroup=" +
                           std::to_string(static_cast<unsigned long long>(is_tiled ? kSharedBytesPerWorkgroup : 0U)));
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    append_note(notes, "barriers_per_workgroup=" + std::to_string(is_tiled ? 1U : 0U));
    append_note(notes, "estimated_global_read_bytes=" +
                           std::to_string(static_cast<unsigned long long>(estimated_global_read_bytes)));
    append_note(notes, "estimated_global_write_bytes=" +
                           std::to_string(static_cast<unsigned long long>(estimated_global_write_bytes)));
    append_note(notes, "validation_mode=exact_uint32");
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
              const PipelineResources& pipeline_resources, ImplementationKind kind, uint32_t radius,
              uint32_t logical_count, bool rounded_to_workgroup_multiple, const std::vector<uint32_t>& reference_values,
              SharedMemoryTilingExperimentOutput& output, bool verbose_progress, std::size_t scratch_size_bytes) {
    const std::string variant_name = make_variant_name(kind, radius);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    const uint32_t source_padded_elements = compute_source_padded_elements(logical_count);
    const auto* src_values = static_cast<const uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped pointers or illegal dispatch size for variant=" << variant_name << ".\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_outputs=" << logical_count << ", source_padded_elements=" << source_padded_elements
                  << ", group_count_x=" << group_count_x << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, radius);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, source_padded_elements) &&
                             validate_output_values(dst_values, reference_values);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_outputs=" << logical_count << ", radius=" << radius
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    const VkDeviceSize estimated_global_read_bytes =
        compute_estimated_global_read_bytes(kind, logical_count, radius, group_count_x);
    const VkDeviceSize estimated_global_write_bytes = compute_estimated_global_write_bytes(logical_count);

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, radius);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, source_padded_elements) &&
                             validate_output_values(dst_values, reference_values);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, kind, radius, logical_count, group_count_x, rounded_to_workgroup_multiple,
                          correctness_pass, dispatch_ok, scratch_size_bytes);

        if (verbose_progress) {
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
            .gbps = compute_effective_gbps(estimated_global_read_bytes, estimated_global_write_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(kind, radius, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_outputs=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

SharedMemoryTilingExperimentOutput
run_shared_memory_tiling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const SharedMemoryTilingExperimentConfig& config) {
    SharedMemoryTilingExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "shared memory tiling experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kImplementationDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
        const auto& descriptor = kImplementationDescriptors[index];
        const std::string user_path =
            descriptor.kind == ImplementationKind::DirectGlobal ? config.direct_shader_path : config.tiled_shader_path;
        shader_paths[index] = VulkanComputeUtils::resolve_shader_path(user_path, descriptor.shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for shared memory tiling variant "
                      << descriptor.implementation_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for shared memory tiling experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = count_resolution.logical_count;
    const VkDeviceSize source_span_bytes = compute_source_span_bytes(logical_count);
    const VkDeviceSize output_span_bytes = compute_output_span_bytes(logical_count);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Direct shader: " << shader_paths[0] << "\n";
        std::cout << "[" << kExperimentId << "] Tiled shader: " << shader_paths[1] << "\n";
        std::cout << "[" << kExperimentId << "] logical_outputs=" << logical_count
                  << ", source_span_bytes=" << source_span_bytes << ", output_span_bytes=" << output_span_bytes
                  << ", scratch_size_bytes=" << config.scratch_size_bytes
                  << ", per_buffer_budget_bytes=" << config.max_buffer_bytes << ", rounded_to_workgroup_multiple="
                  << (count_resolution.rounded_to_workgroup_multiple ? "true" : "false")
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, source_span_bytes, output_span_bytes, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* src_values = static_cast<uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for source or destination buffers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_source_values(src_values, compute_source_padded_elements(logical_count));
    fill_destination_values(dst_values, logical_count);

    std::array<PipelineResources, kImplementationDescriptors.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (const uint32_t radius : kReuseRadii) {
        const std::vector<uint32_t> reference_values = build_reference_values(src_values, logical_count, radius);
        for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
            const auto& descriptor = kImplementationDescriptors[index];
            if (!run_case(context, runner, buffers, pipeline_resources[index], descriptor.kind, radius, logical_count,
                          count_resolution.rounded_to_workgroup_multiple, reference_values, output,
                          config.verbose_progress, config.scratch_size_bytes)) {
                output.all_points_correct = false;
            }
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
