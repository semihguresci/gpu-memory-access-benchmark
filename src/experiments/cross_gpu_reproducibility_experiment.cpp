#include "experiments/cross_gpu_reproducibility_experiment.hpp"

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

constexpr const char* kExperimentId = "45_cross_gpu_reproducibility";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTargetElementCount = 134217728U;
constexpr uint32_t kMinimumElementCount = kWorkgroupSize;
constexpr uint32_t kStrideValue = 8U;
constexpr uint32_t kRandomSeed = 0xA511E9B3U;
constexpr uint32_t kOutputSentinel = 0xDEADBEEFU;

enum class AccessMode : uint32_t {
    CoalescedReference = 0U,
    Stride8Probe = 1U,
    HashedProbe = 2U,
};

struct VariantDescriptor {
    AccessMode mode;
    const char* variant_name;
};

constexpr std::array<VariantDescriptor, 3> kVariantDescriptors = {{
    {AccessMode::CoalescedReference, "coalesced_reference"},
    {AccessMode::Stride8Probe, "stride8_probe"},
    {AccessMode::HashedProbe, "hashed_probe"},
}};

struct BufferResources {
    BufferResource source_buffer{};
    BufferResource output_buffer{};
    void* source_mapped_ptr = nullptr;
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
    uint32_t element_count = 0U;
    uint32_t access_mode = 0U;
    uint32_t stride = 0U;
    uint32_t random_seed = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t hash32(uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

uint32_t source_pattern_value(uint32_t index) {
    return hash32((index * 747796405U) ^ 0x6A09E667U);
}

uint32_t resolve_source_index(uint32_t index, uint32_t element_count, AccessMode mode) {
    if (mode == AccessMode::Stride8Probe) {
        return (index * kStrideValue) % element_count;
    }
    if (mode == AccessMode::HashedProbe) {
        return hash32(index ^ kRandomSeed) % element_count;
    }
    return index;
}

uint32_t transformed_output(uint32_t source_value, uint32_t index) {
    uint32_t state = source_value ^ (index * 0x9E3779B9U);
    state ^= state >> 16U;
    state *= 0x7FEB352DU;
    state ^= state >> 15U;
    state *= 0x846CA68BU;
    state ^= state >> 16U;
    return state;
}

void fill_source_values(uint32_t* source_values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        source_values[index] = source_pattern_value(index);
    }
}

bool validate_source_values(const uint32_t* source_values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (source_values[index] != source_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> build_reference_outputs(const uint32_t* source_values, uint32_t element_count, AccessMode mode) {
    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        const uint32_t source_index = resolve_source_index(index, element_count, mode);
        reference[index] = transformed_output(source_values[source_index], index);
    }
    return reference;
}

uint64_t compute_output_checksum(const uint32_t* output_values, uint32_t element_count) {
    constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = kOffsetBasis;
    for (uint32_t index = 0U; index < element_count; ++index) {
        hash ^= static_cast<uint64_t>(output_values[index]);
        hash *= kPrime;
    }
    return hash;
}

bool validate_outputs(const uint32_t* output_values, const std::vector<uint32_t>& reference, uint32_t element_count) {
    if (output_values == nullptr || reference.size() < element_count) {
        return false;
    }
    return std::equal(reference.begin(), reference.begin() + element_count, output_values);
}

uint32_t determine_element_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t element_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped = std::min({element_capacity, dispatch_capacity, static_cast<uint64_t>(kTargetElementCount)});
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kMinimumElementCount) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_element_count) {
    const std::array<uint32_t, 7> base_sizes = {
        1048576U, 4194304U, 8388608U, 16777216U, 33554432U, 67108864U, 134217728U,
    };
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_element_count) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_element_count) {
        sizes.push_back(max_element_count);
    }
    return sizes;
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint64_t compute_logical_payload_bytes(uint32_t element_count) {
    return static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 2ULL;
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_element_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.source_buffer)) {
        std::cerr << "Failed to create cross-gpu source buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create cross-gpu output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.source_buffer, "cross-gpu source buffer",
                           out_resources.source_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_buffer, "cross-gpu output buffer",
                           out_resources.output_mapped_ptr)) {
        return false;
    }
    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.source_mapped_ptr != nullptr && resources.source_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.source_buffer.memory);
        resources.source_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.source_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo source_info{buffers.source_buffer.buffer, 0U, buffers.source_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    // Shader binding contract:
    //   0 -> deterministic source dataset shared across all access modes
    //   1 -> per-index transformed results for cross-GPU comparison
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, source_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load cross-gpu reproducibility shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create cross-gpu reproducibility descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create cross-gpu reproducibility descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate cross-gpu reproducibility descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create cross-gpu reproducibility pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create cross-gpu reproducibility compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t element_count,
                    AccessMode mode, uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // access_mode selects the source-index mapping:
    //   0 = coalesced reference
    //   1 = fixed stride-8 probe
    //   2 = hashed probe
    const PushConstants constants{
        .element_count = element_count,
        .access_mode = static_cast<uint32_t>(mode),
        .stride = kStrideValue,
        .random_seed = kRandomSeed,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       uint64_t logical_payload_bytes, uint64_t checksum, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "variant=" + std::string(descriptor.variant_name));
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "stride=" + std::to_string(kStrideValue));
    append_note(notes, "random_seed=" + std::to_string(kRandomSeed));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "logical_payload_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "estimated_global_total_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "gbps_mode=logical_payload_bytes");
    append_note(notes, "result_checksum=" + std::to_string(checksum));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, const VariantDescriptor& descriptor, uint32_t element_count,
              uint32_t max_dispatch_groups_x, CrossGpuReproducibilityExperimentOutput& output, bool verbose_progress) {
    const auto* source_values = static_cast<const uint32_t*>(buffers.source_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (source_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    const bool source_valid = validate_source_values(source_values, element_count);
    const std::vector<uint32_t> reference_outputs =
        build_reference_outputs(source_values, element_count, descriptor.mode);
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(element_count);
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        std::fill_n(output_values, element_count, kOutputSentinel);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, element_count, descriptor.mode, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && source_valid && validate_outputs(output_values, reference_outputs, element_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        std::fill_n(output_values, element_count, kOutputSentinel);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, element_count, descriptor.mode, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && source_valid && validate_outputs(output_values, reference_outputs, element_count);
        const uint64_t checksum = compute_output_checksum(output_values, element_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, element_count, logical_payload_bytes, checksum, correctness_pass,
                          dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", checksum=" << checksum << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = 1U,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, 1U, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(logical_payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

} // namespace

CrossGpuReproducibilityExperimentOutput
run_cross_gpu_reproducibility_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                         const CrossGpuReproducibilityExperimentConfig& config) {
    CrossGpuReproducibilityExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "cross-gpu reproducibility experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "45_cross_gpu_reproducibility.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for cross-gpu reproducibility experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for cross-gpu reproducibility experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_element_count
                  << ", span_bytes_per_buffer=" << compute_span_bytes(max_element_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_element_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* source_values = static_cast<uint32_t*>(buffers.source_mapped_ptr);
    if (source_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped source pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_source_values(source_values, max_element_count);

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_element_count);
    for (uint32_t element_count : problem_sizes) {
        for (const auto& descriptor : kVariantDescriptors) {
            if (!run_case(context, runner, pipeline, buffers, descriptor, element_count, max_dispatch_groups_x, output,
                          config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
