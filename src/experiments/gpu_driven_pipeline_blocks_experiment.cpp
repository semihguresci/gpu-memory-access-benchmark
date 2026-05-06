#include "experiments/gpu_driven_pipeline_blocks_experiment.hpp"

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

constexpr const char* kExperimentId = "44_gpu_driven_pipeline_blocks";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTargetElementCount = 67108864U;
constexpr uint32_t kMinimumElementCount = kWorkgroupSize;
constexpr uint32_t kVisibilityThresholdMilli = 650U;
constexpr uint32_t kBucketCount = 64U;
constexpr uint32_t kBucketSentinel = 0xFFFFFFFFU;
constexpr uint32_t kVisibilitySentinel = 0xCDCDCDCDU;
constexpr uint32_t kOutputSentinel = 0xDEADBEEFU;

enum class VariantKind : uint32_t {
    StagedThreeDispatch = 0U,
    FusedSingleDispatch = 1U,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    uint32_t dispatch_count;
    uint64_t estimated_total_bytes_per_element;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::StagedThreeDispatch, "staged_three_dispatch", 3U, 36ULL},
    {VariantKind::FusedSingleDispatch, "fused_single_dispatch", 1U, 16ULL},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource visibility_buffer{};
    BufferResource bucket_buffer{};
    BufferResource output_buffer{};
    void* input_mapped_ptr = nullptr;
    void* visibility_mapped_ptr = nullptr;
    void* bucket_mapped_ptr = nullptr;
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
    uint32_t mode = 0U;
    uint32_t visibility_threshold_milli = 0U;
    uint32_t bucket_count = 0U;
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

uint32_t input_pattern_value(uint32_t index) {
    return hash32((index * 2654435761U) ^ 0x31415926U);
}

bool is_visible(uint32_t input_value, uint32_t visibility_threshold_milli) {
    const uint32_t metric = hash32(input_value ^ 0x9E3779B9U) % 1000U;
    return metric < visibility_threshold_milli;
}

uint32_t bucket_for_value(uint32_t input_value, uint32_t bucket_count) {
    return hash32(input_value ^ 0x85EBCA6BU) % bucket_count;
}

uint32_t emit_token(uint32_t input_value, uint32_t bucket) {
    const uint32_t payload = hash32(input_value ^ 0xC2B2AE35U);
    return ((bucket & 0xFFU) << 24U) ^ (payload & 0x00FFFFFFU);
}

void fill_input_values(uint32_t* input_values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        input_values[index] = input_pattern_value(index);
    }
}

bool validate_input_values(const uint32_t* input_values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (input_values[index] != input_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

void build_reference_outputs(const uint32_t* input_values, uint32_t element_count, uint32_t visibility_threshold_milli,
                             uint32_t bucket_count, std::vector<uint32_t>& visibility_reference,
                             std::vector<uint32_t>& bucket_reference, std::vector<uint32_t>& output_reference) {
    visibility_reference.assign(element_count, 0U);
    bucket_reference.assign(element_count, kBucketSentinel);
    output_reference.assign(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        const uint32_t input_value = input_values[index];
        const bool visible = is_visible(input_value, visibility_threshold_milli);
        visibility_reference[index] = visible ? 1U : 0U;
        if (visible) {
            const uint32_t bucket = bucket_for_value(input_value, bucket_count);
            bucket_reference[index] = bucket;
            output_reference[index] = emit_token(input_value, bucket);
        }
    }
}

bool validate_outputs(const uint32_t* visibility_values, const uint32_t* bucket_values, const uint32_t* output_values,
                      const std::vector<uint32_t>& visibility_reference, const std::vector<uint32_t>& bucket_reference,
                      const std::vector<uint32_t>& output_reference, uint32_t element_count) {
    if (visibility_values == nullptr || bucket_values == nullptr || output_values == nullptr ||
        visibility_reference.size() < element_count || bucket_reference.size() < element_count ||
        output_reference.size() < element_count) {
        return false;
    }
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (visibility_values[index] != visibility_reference[index] ||
            bucket_values[index] != bucket_reference[index] || output_values[index] != output_reference[index]) {
            return false;
        }
    }
    return true;
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
    const std::array<uint32_t, 6> base_sizes = {1048576U, 4194304U, 8388608U, 16777216U, 33554432U, 67108864U};
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
                                out_resources.input_buffer)) {
        std::cerr << "Failed to create GPU-driven input buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.visibility_buffer)) {
        std::cerr << "Failed to create GPU-driven visibility buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.bucket_buffer)) {
        std::cerr << "Failed to create GPU-driven bucket buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.visibility_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create GPU-driven output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.bucket_buffer);
        destroy_buffer_resource(context.device(), out_resources.visibility_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "gpu-driven input buffer",
                           out_resources.input_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.visibility_buffer, "gpu-driven visibility buffer",
                           out_resources.visibility_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.bucket_buffer, "gpu-driven bucket buffer",
                           out_resources.bucket_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_buffer, "gpu-driven output buffer",
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
    if (resources.bucket_mapped_ptr != nullptr && resources.bucket_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.bucket_buffer.memory);
        resources.bucket_mapped_ptr = nullptr;
    }
    if (resources.visibility_mapped_ptr != nullptr && resources.visibility_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.visibility_buffer.memory);
        resources.visibility_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.bucket_buffer);
    destroy_buffer_resource(context.device(), resources.visibility_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo visibility_info{buffers.visibility_buffer.buffer, 0U, buffers.visibility_buffer.size};
    const VkDescriptorBufferInfo bucket_info{buffers.bucket_buffer.buffer, 0U, buffers.bucket_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    // Shader binding contract:
    //   0 -> canonical input stream
    //   1 -> stage-A visibility flags
    //   2 -> stage-B bucket ids or sentinel
    //   3 -> final emitted tokens
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, visibility_info},
                                                          {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bucket_info},
                                                          {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load GPU-driven pipeline shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create GPU-driven pipeline descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create GPU-driven pipeline descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate GPU-driven pipeline descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create GPU-driven pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create GPU-driven pipeline compute pipeline.\n";
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

void record_compute_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         1U, &barrier, 0U, nullptr, 0U, nullptr);
}

double run_variant_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t element_count,
                            VariantKind variant_kind, uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);

        if (variant_kind == VariantKind::StagedThreeDispatch) {
            // Staged path replays the shader three times:
            //   0 = visibility classification -> binding 1
            //   1 = bucket assignment -> binding 2
            //   2 = token emission -> binding 3
            for (uint32_t mode = 0U; mode < 3U; ++mode) {
                const PushConstants constants{
                    .element_count = element_count,
                    .mode = mode,
                    .visibility_threshold_milli = kVisibilityThresholdMilli,
                    .bucket_count = kBucketCount,
                };
                vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                   static_cast<uint32_t>(sizeof(constants)), &constants);
                vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
                if (mode < 2U) {
                    record_compute_barrier(command_buffer);
                }
            }
            return;
        }

        // Fused path uses mode 3 to execute the whole pipeline in one dispatch.
        const PushConstants constants{
            .element_count = element_count,
            .mode = 3U,
            .visibility_threshold_milli = kVisibilityThresholdMilli,
            .bucket_count = kBucketCount,
        };
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void reset_outputs(uint32_t* visibility_values, uint32_t* bucket_values, uint32_t* output_values,
                   uint32_t element_count) {
    std::fill_n(visibility_values, element_count, kVisibilitySentinel);
    std::fill_n(bucket_values, element_count, kBucketSentinel);
    std::fill_n(output_values, element_count, kOutputSentinel);
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       uint64_t logical_payload_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "variant=" + std::string(descriptor.variant_name));
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "dispatches_per_iteration=" + std::to_string(descriptor.dispatch_count));
    append_note(notes, "visibility_threshold_milli=" + std::to_string(kVisibilityThresholdMilli));
    append_note(notes, "bucket_count=" + std::to_string(kBucketCount));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "logical_payload_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "estimated_global_total_bytes=" + std::to_string(descriptor.estimated_total_bytes_per_element *
                                                                        static_cast<uint64_t>(element_count)));
    append_note(notes, "gbps_mode=logical_payload_bytes");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, const VariantDescriptor& descriptor, uint32_t element_count,
              uint32_t max_dispatch_groups_x, GpuDrivenPipelineBlocksExperimentOutput& output, bool verbose_progress) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* visibility_values = static_cast<uint32_t*>(buffers.visibility_mapped_ptr);
    auto* bucket_values = static_cast<uint32_t*>(buffers.bucket_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || visibility_values == nullptr || bucket_values == nullptr ||
        output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    std::vector<uint32_t> visibility_reference;
    std::vector<uint32_t> bucket_reference;
    std::vector<uint32_t> output_reference;
    build_reference_outputs(input_values, element_count, kVisibilityThresholdMilli, kBucketCount, visibility_reference,
                            bucket_reference, output_reference);

    const bool input_valid = validate_input_values(input_values, element_count);
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(element_count);
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_outputs(visibility_values, bucket_values, output_values, element_count);
        const double dispatch_ms =
            run_variant_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && input_valid &&
                validate_outputs(visibility_values, bucket_values, output_values, visibility_reference,
                                 bucket_reference, output_reference, element_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        reset_outputs(visibility_values, bucket_values, output_values, element_count);
        const double dispatch_ms =
            run_variant_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid &&
            validate_outputs(visibility_values, bucket_values, output_values, visibility_reference, bucket_reference,
                             output_reference, element_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, element_count, logical_payload_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = descriptor.dispatch_count,
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

GpuDrivenPipelineBlocksExperimentOutput
run_gpu_driven_pipeline_blocks_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                          const GpuDrivenPipelineBlocksExperimentConfig& config) {
    GpuDrivenPipelineBlocksExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "gpu-driven pipeline blocks experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "44_gpu_driven_pipeline_blocks.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for gpu-driven pipeline blocks experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for gpu-driven pipeline blocks experiment.\n";
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

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    if (input_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped input pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_input_values(input_values, max_element_count);

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
