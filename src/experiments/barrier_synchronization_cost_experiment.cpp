#include "experiments/barrier_synchronization_cost_experiment.hpp"

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

constexpr const char* kExperimentId = "20_barrier_synchronization_cost";
constexpr uint32_t kLocalSizeX = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTotalPhases = 8U;
constexpr uint32_t kSeedSalt = 0xA5A5A5A5U;
constexpr VkDeviceSize kBytesPerElement = static_cast<VkDeviceSize>(sizeof(uint32_t) * 2U);
constexpr std::array<uint32_t, 5> kBarrierIntervalsPhases = {0U, 1U, 2U, 4U, 8U};

enum class PlacementKind : std::uint8_t {
    FlatLoop = 0U,
    TiledRegions = 1U,
};

struct VariantDescriptor {
    PlacementKind placement = PlacementKind::FlatLoop;
    const char* placement_name = "";
    const char* shader_filename = "";
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {PlacementKind::FlatLoop, "flat_loop", "20_barrier_synchronization_cost_flat.comp.spv"},
    {PlacementKind::TiledRegions, "tiled_regions", "20_barrier_synchronization_cost_tiled.comp.spv"},
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
    uint32_t logical_count = 0U;
    uint32_t barrier_interval_phases = 0U;
    uint32_t total_phases = 0U;
    uint32_t seed_salt = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_variant_name(PlacementKind placement, uint32_t barrier_interval_phases) {
    const char* placement_name = placement == PlacementKind::FlatLoop ? "flat_loop" : "tiled_regions";
    return std::string(placement_name) + "_b" + std::to_string(barrier_interval_phases);
}

std::string make_case_name(PlacementKind placement, uint32_t barrier_interval_phases, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(placement, barrier_interval_phases) + "_outputs_" +
           std::to_string(logical_count);
}

uint32_t source_pattern_value(uint32_t index) {
    return (index * 1103515245U + 12345U) ^ 0x9E3779B9U;
}

void fill_source_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_destination_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, 0xA5A5A5A5U);
}

bool validate_source_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != source_pattern_value(index)) {
            return false;
        }
    }

    return true;
}

uint32_t rotl32(uint32_t value, uint32_t shift) {
    const uint32_t effective_shift = shift & 31U;
    return (value << effective_shift) | (value >> ((32U - effective_shift) & 31U));
}

uint32_t mix_value(uint32_t value, uint32_t phase_index, uint32_t local_id, uint32_t group_id, uint32_t tile_index) {
    const uint32_t phase_salt = 0x9E3779B9U * (phase_index + 1U);
    const uint32_t local_salt = 0x85EBCA6BU * (local_id + 1U);
    const uint32_t group_salt = 0xC2B2AE35U * (group_id + 1U);
    const uint32_t tile_salt = 0x27D4EB2DU * (tile_index + 1U);
    value ^= phase_salt + local_salt;
    value += group_salt ^ tile_salt;
    value = rotl32(value, (phase_index % 13U) + 5U);
    value ^= value >> 11U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    return value;
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (output_values[index] != reference_values[index]) {
            return false;
        }
    }

    return true;
}

std::vector<uint32_t> build_reference_values(const uint32_t* src_values, uint32_t logical_count,
                                             uint32_t barrier_interval_phases) {
    std::vector<uint32_t> reference_values(logical_count, 0U);
    const uint32_t tile_phase_count = barrier_interval_phases == 0U ? kTotalPhases : barrier_interval_phases;
    const uint32_t tile_count = barrier_interval_phases == 0U ? 1U : (kTotalPhases / tile_phase_count);
    const uint32_t group_count_x = logical_count / kLocalSizeX;

    for (uint32_t group_id = 0U; group_id < group_count_x; ++group_id) {
        std::array<uint32_t, kLocalSizeX> values{};
        const uint32_t group_base = group_id * kLocalSizeX;
        for (uint32_t local_id = 0U; local_id < kLocalSizeX; ++local_id) {
            values[local_id] = src_values[group_base + local_id];
        }

        uint32_t phase_index = 0U;
        for (uint32_t tile_index = 0U; tile_index < tile_count; ++tile_index) {
            for (uint32_t tile_phase = 0U; tile_phase < tile_phase_count; ++tile_phase) {
                for (uint32_t local_id = 0U; local_id < kLocalSizeX; ++local_id) {
                    values[local_id] = mix_value(values[local_id], phase_index, local_id, group_id, tile_index);
                }
                ++phase_index;
            }

            if (barrier_interval_phases != 0U) {
                const uint32_t shared_seed =
                    values[0U] ^ (kSeedSalt + (group_id * 0x9E3779B9U) + (tile_index * 0x85EBCA6BU));
                for (uint32_t local_id = 0U; local_id < kLocalSizeX; ++local_id) {
                    values[local_id] ^= shared_seed + local_id + group_id + phase_index;
                }
            }
        }

        for (uint32_t local_id = 0U; local_id < kLocalSizeX; ++local_id) {
            reference_values[group_base + local_id] = values[local_id];
        }
    }

    return reference_values;
}

CountResolution determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    CountResolution resolution{};
    const uint64_t buffer_limited_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_elements = static_cast<uint64_t>(max_dispatch_groups_x) * kLocalSizeX;
    const uint64_t effective_elements = std::min({buffer_limited_elements, dispatch_limited_elements,
                                                  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});

    if (effective_elements < kLocalSizeX) {
        return resolution;
    }

    const uint64_t rounded_elements = effective_elements - (effective_elements % kLocalSizeX);
    resolution.logical_count = static_cast<uint32_t>(rounded_elements);
    resolution.rounded_to_workgroup_multiple = rounded_elements != effective_elements;
    return resolution;
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize buffer_size, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.src_buffer)) {
        std::cerr << "Failed to create barrier synchronization source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create barrier synchronization destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "barrier synchronization source buffer",
                           out_resources.src_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "barrier synchronization destination buffer",
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
        std::cerr << "Failed to load barrier synchronization shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create barrier synchronization descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create barrier synchronization descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate barrier synchronization descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create barrier synchronization pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create barrier synchronization compute pipeline.\n";
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
                    uint32_t barrier_interval_phases) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kLocalSizeX);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        logical_count,
        barrier_interval_phases,
        kTotalPhases,
        kSeedSalt,
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

void record_case_notes(std::string& notes, PlacementKind placement, uint32_t barrier_interval_phases,
                       uint32_t logical_count, uint32_t group_count_x, bool rounded_to_workgroup_multiple,
                       bool correctness_pass, bool dispatch_ok, std::size_t scratch_size_bytes) {
    const uint32_t sync_points = barrier_interval_phases == 0U ? 0U : (kTotalPhases / barrier_interval_phases);
    const uint32_t tile_phase_count = barrier_interval_phases == 0U ? kTotalPhases : barrier_interval_phases;
    const uint32_t tile_count = barrier_interval_phases == 0U ? 1U : (kTotalPhases / tile_phase_count);
    const bool is_flat = placement == PlacementKind::FlatLoop;

    append_note(notes, std::string("placement=") + (is_flat ? "flat_loop" : "tiled_regions"));
    append_note(notes, "barrier_interval_phases=" + std::to_string(barrier_interval_phases));
    append_note(notes, "sync_points=" + std::to_string(sync_points));
    append_note(notes, "work_phases=" + std::to_string(kTotalPhases));
    append_note(notes, "tile_phase_count=" + std::to_string(tile_phase_count));
    append_note(notes, "tile_count=" + std::to_string(tile_count));
    append_note(notes, "local_size_x=" + std::to_string(kLocalSizeX));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes,
                "shared_bytes_per_workgroup=" + std::to_string(static_cast<unsigned long long>(sizeof(uint32_t))));
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    append_note(notes, "validation_mode=exact_uint32");
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
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
              const PipelineResources& pipeline_resources, PlacementKind placement, uint32_t barrier_interval_phases,
              uint32_t logical_count, bool rounded_to_workgroup_multiple, const std::vector<uint32_t>& reference_values,
              BarrierSynchronizationCostExperimentOutput& output, bool verbose_progress,
              std::size_t scratch_size_bytes) {
    const std::string variant_name = make_variant_name(placement, barrier_interval_phases);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kLocalSizeX);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for variant " << variant_name
                  << ".\n";
        return false;
    }

    const auto* src_values = static_cast<const uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for variant " << variant_name << ".\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_outputs=" << logical_count << ", barrier_interval_phases=" << barrier_interval_phases
                  << ", group_count_x=" << group_count_x << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, barrier_interval_phases);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, logical_count) &&
                             validate_output_values(dst_values, reference_values);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_outputs=" << logical_count
                      << ", barrier_interval_phases=" << barrier_interval_phases
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    const VkDeviceSize total_bytes = static_cast<VkDeviceSize>(logical_count) * kBytesPerElement;

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, barrier_interval_phases);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, logical_count) &&
                             validate_output_values(dst_values, reference_values);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, placement, barrier_interval_phases, logical_count, group_count_x,
                          rounded_to_workgroup_multiple, correctness_pass, dispatch_ok, scratch_size_bytes);

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
            .gbps = compute_effective_gbps_from_bytes(static_cast<uint64_t>(total_bytes), dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary = BenchmarkRunner::summarize_samples(
        make_case_name(placement, barrier_interval_phases, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_outputs=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

BarrierSynchronizationCostExperimentOutput
run_barrier_synchronization_cost_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                            const BarrierSynchronizationCostExperimentConfig& config) {
    BarrierSynchronizationCostExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "barrier synchronization cost experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        const auto& descriptor = kVariantDescriptors[index];
        const std::string user_path =
            descriptor.placement == PlacementKind::FlatLoop ? config.flat_shader_path : config.tiled_shader_path;
        shader_paths[index] = VulkanComputeUtils::resolve_shader_path(user_path, descriptor.shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for barrier synchronization variant "
                      << descriptor.placement_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for barrier synchronization cost experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = count_resolution.logical_count;
    const VkDeviceSize span_bytes = compute_span_bytes(logical_count);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Flat shader: " << shader_paths[0] << "\n";
        std::cout << "[" << kExperimentId << "] Tiled shader: " << shader_paths[1] << "\n";
        std::cout << "[" << kExperimentId << "] logical_outputs=" << logical_count
                  << ", span_bytes_per_buffer=" << span_bytes << ", scratch_size_bytes=" << config.scratch_size_bytes
                  << ", per_buffer_budget_bytes=" << config.max_buffer_bytes << ", work_phases=" << kTotalPhases
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, span_bytes, buffers)) {
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

    fill_source_values(src_values, logical_count);
    fill_destination_values(dst_values, logical_count);

    std::array<PipelineResources, kVariantDescriptors.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (const uint32_t barrier_interval_phases : kBarrierIntervalsPhases) {
        const std::vector<uint32_t> reference_values =
            build_reference_values(src_values, logical_count, barrier_interval_phases);
        for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
            const auto& descriptor = kVariantDescriptors[index];
            if (!run_case(context, runner, buffers, pipeline_resources[index], descriptor.placement,
                          barrier_interval_phases, logical_count, count_resolution.rounded_to_workgroup_multiple,
                          reference_values, output, config.verbose_progress, config.scratch_size_bytes)) {
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
