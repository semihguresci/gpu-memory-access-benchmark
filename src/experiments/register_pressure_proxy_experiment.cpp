#include "experiments/register_pressure_proxy_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

constexpr const char* kExperimentId = "18_register_pressure_proxy";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kRounds = 8U;
constexpr uint32_t kMaxTempCount = 32U;
constexpr uint32_t kSourceMultiplier = 0x9E3779B9U;
constexpr uint32_t kSourceOffset = 0x6D2B79F5U;
constexpr uint32_t kIndexMultiplier = 0x85EBCA6BU;
constexpr uint32_t kLaneMultiplier = 0x7F4A7C15U;
constexpr uint32_t kRoundMultiplier = 0xC2B2AE35U;
constexpr uint32_t kResultMixMultiplier = 0xD2511F53U;
constexpr uint32_t kResultXorMultiplier = 0xA24BAED5U;
constexpr uint32_t kResultIndexMultiplier = 0x27D4EB2DU;
constexpr uint32_t kDestinationSentinelValue = 0xA5A5A5A5U;

struct VariantDescriptor {
    const char* variant_name = "";
    const char* shader_filename = "";
    uint32_t temp_count = 0U;
};

constexpr std::array<VariantDescriptor, 4> kVariants = {{
    {"temp_4", "18_register_pressure_proxy_temp4.comp.spv", 4U},
    {"temp_8", "18_register_pressure_proxy_temp8.comp.spv", 8U},
    {"temp_16", "18_register_pressure_proxy_temp16.comp.spv", 16U},
    {"temp_32", "18_register_pressure_proxy_temp32.comp.spv", 32U},
}};

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
    uint32_t element_count = 0U;
    uint32_t rounds = 0U;
    uint32_t temp_count = 0U;
    uint32_t reserved = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t) * 4U);

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string make_case_name(const std::string& variant_name, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + variant_name + "_elements_" + std::to_string(logical_count);
}

uint32_t source_pattern_value(uint32_t index) {
    return ((index + 1U) * kSourceMultiplier) ^ kSourceOffset;
}

uint32_t rotate_left(uint32_t value, uint32_t shift) {
    return std::rotl(value, static_cast<int>(shift & 31U));
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

bool validate_destination_values(const uint32_t* values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (values[index] != reference_values[index]) {
            return false;
        }
    }
    return true;
}

uint32_t transform_value(uint32_t seed, uint32_t index, uint32_t temp_count) {
    std::array<uint32_t, kMaxTempCount> state{};
    for (uint32_t lane = 0U; lane < temp_count; ++lane) {
        uint32_t lane_seed = seed ^ ((lane + 1U) * kLaneMultiplier);
        lane_seed += index * kIndexMultiplier;
        lane_seed ^= rotate_left(seed + ((lane + 1U) * kLaneMultiplier), (lane * 5U + 1U) & 31U);
        state[lane] = lane_seed;
    }

    for (uint32_t round = 0U; round < kRounds; ++round) {
        for (uint32_t lane = 0U; lane < temp_count; ++lane) {
            const uint32_t left = state[(lane + temp_count - 1U) % temp_count];
            const uint32_t right = state[(lane + 1U) % temp_count];
            const uint32_t current = state[lane];
            const uint32_t mixed = rotate_left(
                current + (left ^ (round * kRoundMultiplier)) + ((lane + 1U) * kLaneMultiplier), (lane + round) & 31U);
            state[lane] = (mixed ^ right) + kResultMixMultiplier + ((lane + 1U) * kResultXorMultiplier);
        }
    }

    uint32_t result =
        seed ^ (index * kResultIndexMultiplier) ^ (temp_count * kIndexMultiplier) ^ (kRounds * kSourceMultiplier);
    for (uint32_t lane = 0U; lane < temp_count; ++lane) {
        result ^= rotate_left(state[lane] + ((lane + 1U) * kResultMixMultiplier), (lane + temp_count) & 31U);
        result += state[lane] ^ ((lane + 1U) * kResultXorMultiplier);
    }

    return result;
}

std::vector<uint32_t> build_reference_values(const uint32_t* src_values, uint32_t element_count, uint32_t temp_count) {
    std::vector<uint32_t> reference_values(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        reference_values[index] = transform_value(src_values[index], index, temp_count);
    }
    return reference_values;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t buffer_limited_count = max_buffer_bytes / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count_u64 = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_count_u64 < kWorkgroupSize) {
        return 0U;
    }

    return static_cast<uint32_t>(effective_count_u64 - (effective_count_u64 % kWorkgroupSize));
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_effective_payload_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t) * 2U);
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.src_buffer)) {
        std::cerr << "Failed to create register pressure proxy source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create register pressure proxy destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "register pressure proxy source buffer",
                           out_resources.src_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "register pressure proxy destination buffer",
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
        std::cerr << "Failed to load register pressure proxy shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create register pressure proxy descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create register pressure proxy descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate register pressure proxy descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create register pressure proxy pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create register pressure proxy compute pipeline.\n";
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
                    uint32_t temp_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        .element_count = logical_count,
        .rounds = kRounds,
        .temp_count = temp_count,
        .reserved = 0U,
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

void record_case_notes(std::string& notes, const VariantDescriptor& variant, uint32_t logical_count,
                       uint32_t group_count_x, VkDeviceSize span_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("variant=") + variant.variant_name);
    append_note(notes, "temporary_count=" + std::to_string(variant.temp_count));
    append_note(notes, "round_count=" + std::to_string(kRounds));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "source_span_bytes=" + std::to_string(static_cast<unsigned long long>(span_bytes)));
    append_note(notes, "destination_span_bytes=" + std::to_string(static_cast<unsigned long long>(span_bytes)));
    append_note(notes, "payload_bytes_per_element=" + std::to_string(sizeof(uint32_t) * 2U));
    append_note(notes, "register_pressure_proxy=true");
    append_note(notes, "validation_mode=exact_uint32");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, const VariantDescriptor& variant, uint32_t logical_count,
              const std::vector<uint32_t>& reference_values, RegisterPressureProxyExperimentOutput& output,
              bool verbose_progress) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    const VkDeviceSize span_bytes = compute_span_bytes(logical_count);
    const VkDeviceSize payload_bytes = compute_effective_payload_bytes(logical_count);
    auto* src_values = static_cast<uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped pointers or illegal dispatch size for variant=" << variant.variant_name << ".\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant.variant_name
                  << ", logical_elements=" << logical_count << ", temp_count=" << variant.temp_count
                  << ", group_count_x=" << group_count_x << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_source_values(src_values, logical_count);
        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, variant.temp_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, logical_count) &&
                             validate_destination_values(dst_values, reference_values);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant.variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant.variant_name
                      << ", logical_elements=" << logical_count << ", temp_count=" << variant.temp_count
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_source_values(src_values, logical_count);
        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, variant.temp_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_source_values(src_values, logical_count) &&
                             validate_destination_values(dst_values, reference_values);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, variant, logical_count, group_count_x, span_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant.variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant.variant_name,
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

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(variant.variant_name, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant.variant_name
                  << ", logical_elements=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

RegisterPressureProxyExperimentOutput
run_register_pressure_proxy_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const RegisterPressureProxyExperimentConfig& config) {
    RegisterPressureProxyExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "register pressure proxy experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariants.size()> shader_paths;
    for (std::size_t index = 0; index < kVariants.size(); ++index) {
        shader_paths[index] =
            VulkanComputeUtils::resolve_shader_path(config.shader_path, kVariants[index].shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for register pressure proxy variant "
                      << kVariants[index].variant_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for register pressure proxy experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const VkDeviceSize span_bytes = compute_span_bytes(logical_count);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] logical_elements=" << logical_count
                  << ", source_span_bytes=" << span_bytes << ", destination_span_bytes=" << span_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
        for (std::size_t index = 0; index < kVariants.size(); ++index) {
            std::cout << "[" << kExperimentId << "] Shader " << kVariants[index].variant_name << ": "
                      << shader_paths[index] << "\n";
        }
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

    std::array<PipelineResources, kVariants.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kVariants.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (std::size_t index = 0; index < kVariants.size(); ++index) {
        const VariantDescriptor& variant = kVariants[index];
        const std::vector<uint32_t> reference_values =
            build_reference_values(src_values, logical_count, variant.temp_count);
        if (!run_case(context, runner, buffers, pipeline_resources[index], variant, logical_count, reference_values,
                      output, verbose_progress)) {
            output.all_points_correct = false;
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
