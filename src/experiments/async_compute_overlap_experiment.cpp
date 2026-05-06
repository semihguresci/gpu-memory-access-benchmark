#include "experiments/async_compute_overlap_experiment.hpp"

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

constexpr const char* kExperimentId = "42_async_compute_overlap";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTargetElementCount = 67108864U;
constexpr uint32_t kMinimumElementCount = kWorkgroupSize;

enum class VariantKind : uint32_t {
    SerialNoOverlap = 0U,
    FusedOverlapProxy = 1U,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    uint32_t dispatch_count;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::SerialNoOverlap, "serial_no_overlap", 2U},
    {VariantKind::FusedOverlapProxy, "fused_overlap_proxy", 1U},
}};

struct BufferResources {
    BufferResource input_a{};
    BufferResource input_b{};
    BufferResource output_a{};
    BufferResource output_b{};
    void* input_a_mapped_ptr = nullptr;
    void* input_b_mapped_ptr = nullptr;
    void* output_a_mapped_ptr = nullptr;
    void* output_b_mapped_ptr = nullptr;
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
    uint32_t mode = 0U; // 0=pass_a, 1=pass_b, 2=fused
    uint32_t reserved0 = 0U;
    uint32_t reserved1 = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t pattern_a(uint32_t index) {
    uint32_t value = index * 747796405U + 2891336453U;
    value ^= value >> 16U;
    value *= 2246822519U;
    value ^= value >> 13U;
    return value;
}

uint32_t pattern_b(uint32_t index) {
    uint32_t value = (index ^ 0x9E3779B9U) * 1597334677U + 3812015801U;
    value ^= value >> 15U;
    value *= 3266489917U;
    value ^= value >> 16U;
    return value;
}

uint32_t transform_a(uint32_t value) {
    uint32_t state = value;
    state ^= state >> 16U;
    state *= 0x7FEB352DU;
    state ^= state >> 15U;
    state *= 0x846CA68BU;
    state ^= state >> 16U;
    return state;
}

uint32_t transform_b(uint32_t value) {
    uint32_t state = value;
    state ^= (state << 13U);
    state ^= (state >> 17U);
    state ^= (state << 5U);
    return state * 1664525U + 1013904223U;
}

void fill_input_values(uint32_t* values_a, uint32_t* values_b, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values_a[index] = pattern_a(index);
        values_b[index] = pattern_b(index);
    }
}

bool validate_input_values(const uint32_t* values_a, const uint32_t* values_b, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values_a[index] != pattern_a(index) || values_b[index] != pattern_b(index)) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> build_reference_output_a(const uint32_t* input_a, uint32_t element_count) {
    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        reference[index] = transform_a(input_a[index]);
    }
    return reference;
}

std::vector<uint32_t> build_reference_output_b(const uint32_t* input_b, uint32_t element_count) {
    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        reference[index] = transform_b(input_b[index]);
    }
    return reference;
}

bool validate_output_values(const uint32_t* output_a, const uint32_t* output_b,
                            const std::vector<uint32_t>& reference_a, const std::vector<uint32_t>& reference_b,
                            uint32_t element_count) {
    if (output_a == nullptr || output_b == nullptr || reference_a.size() < element_count ||
        reference_b.size() < element_count) {
        return false;
    }
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (output_a[index] != reference_a[index] || output_b[index] != reference_b[index]) {
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
    return static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 4ULL;
}

bool create_buffer_resources(VulkanContext& context, uint32_t element_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_a)) {
        std::cerr << "Failed to create async-overlap input_a buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_b)) {
        std::cerr << "Failed to create async-overlap input_b buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_a);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_a)) {
        std::cerr << "Failed to create async-overlap output_a buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_b);
        destroy_buffer_resource(context.device(), out_resources.input_a);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_b)) {
        std::cerr << "Failed to create async-overlap output_b buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_a);
        destroy_buffer_resource(context.device(), out_resources.input_b);
        destroy_buffer_resource(context.device(), out_resources.input_a);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_a, "async-overlap input_a", out_resources.input_a_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.input_b, "async-overlap input_b", out_resources.input_b_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_a, "async-overlap output_a",
                           out_resources.output_a_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.output_b, "async-overlap output_b",
                           out_resources.output_b_mapped_ptr)) {
        return false;
    }
    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.output_b_mapped_ptr != nullptr && resources.output_b.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_b.memory);
        resources.output_b_mapped_ptr = nullptr;
    }
    if (resources.output_a_mapped_ptr != nullptr && resources.output_a.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_a.memory);
        resources.output_a_mapped_ptr = nullptr;
    }
    if (resources.input_b_mapped_ptr != nullptr && resources.input_b.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_b.memory);
        resources.input_b_mapped_ptr = nullptr;
    }
    if (resources.input_a_mapped_ptr != nullptr && resources.input_a.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_a.memory);
        resources.input_a_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.output_b);
    destroy_buffer_resource(context.device(), resources.output_a);
    destroy_buffer_resource(context.device(), resources.input_b);
    destroy_buffer_resource(context.device(), resources.input_a);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_a_info{buffers.input_a.buffer, 0U, buffers.input_a.size};
    const VkDescriptorBufferInfo input_b_info{buffers.input_b.buffer, 0U, buffers.input_b.size};
    const VkDescriptorBufferInfo output_a_info{buffers.output_a.buffer, 0U, buffers.output_a.size};
    const VkDescriptorBufferInfo output_b_info{buffers.output_b.buffer, 0U, buffers.output_b.size};

    // Shader binding contract:
    //   0 -> input stream for pass A
    //   1 -> input stream for pass B
    //   2 -> output stream for pass A
    //   3 -> output stream for pass B
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_a_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_b_info},
                                                          {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_a_info},
                                                          {3U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_b_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load async-overlap shader module: " << shader_path << "\n";
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
        std::cerr << "Failed to create async-overlap descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create async-overlap descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate async-overlap descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create async-overlap pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create async-overlap compute pipeline.\n";
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

        if (variant_kind == VariantKind::SerialNoOverlap) {
            // Staged path issues two separate dispatches:
            //   mode 0 = pass A only
            //   mode 1 = pass B only
            const PushConstants pass_a{
                .element_count = element_count,
                .mode = 0U,
                .reserved0 = 0U,
                .reserved1 = 0U,
            };
            vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(pass_a)), &pass_a);
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);

            record_compute_barrier(command_buffer);

            const PushConstants pass_b{
                .element_count = element_count,
                .mode = 1U,
                .reserved0 = 0U,
                .reserved1 = 0U,
            };
            vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(pass_b)), &pass_b);
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
            return;
        }

        // Fused path uses mode 2 so one dispatch produces both outputs.
        const PushConstants fused{
            .element_count = element_count,
            .mode = 2U,
            .reserved0 = 0U,
            .reserved1 = 0U,
        };
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(fused)), &fused);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void reset_outputs(uint32_t* output_a, uint32_t* output_b, uint32_t element_count) {
    std::fill_n(output_a, element_count, 0U);
    std::fill_n(output_b, element_count, 0U);
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       uint64_t logical_payload_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "variant=" + std::string(descriptor.variant_name));
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "dispatches_per_sample=" + std::to_string(descriptor.dispatch_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "throughput_scope=logical_elements");
    append_note(notes, "logical_payload_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "estimated_global_total_bytes=" + std::to_string(logical_payload_bytes));
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
              uint32_t max_dispatch_groups_x, AsyncComputeOverlapExperimentOutput& output, bool verbose_progress) {
    const auto* input_a = static_cast<const uint32_t*>(buffers.input_a_mapped_ptr);
    const auto* input_b = static_cast<const uint32_t*>(buffers.input_b_mapped_ptr);
    auto* output_a = static_cast<uint32_t*>(buffers.output_a_mapped_ptr);
    auto* output_b = static_cast<uint32_t*>(buffers.output_b_mapped_ptr);
    if (input_a == nullptr || input_b == nullptr || output_a == nullptr || output_b == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    const bool input_valid = validate_input_values(input_a, input_b, element_count);
    const std::vector<uint32_t> reference_a = build_reference_output_a(input_a, element_count);
    const std::vector<uint32_t> reference_b = build_reference_output_b(input_b, element_count);
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(element_count);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_outputs(output_a, output_b, element_count);
        const double dispatch_ms =
            run_variant_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && input_valid &&
                validate_output_values(output_a, output_b, reference_a, reference_b, element_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        reset_outputs(output_a, output_b, element_count);
        const double dispatch_ms =
            run_variant_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid &&
            validate_output_values(output_a, output_b, reference_a, reference_b, element_count);
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

AsyncComputeOverlapExperimentOutput
run_async_compute_overlap_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                     const AsyncComputeOverlapExperimentConfig& config) {
    AsyncComputeOverlapExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "async compute overlap experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "42_async_compute_overlap.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for async compute overlap experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for async compute overlap experiment.\n";
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

    auto* input_a = static_cast<uint32_t*>(buffers.input_a_mapped_ptr);
    auto* input_b = static_cast<uint32_t*>(buffers.input_b_mapped_ptr);
    if (input_a == nullptr || input_b == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped input pointers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_input_values(input_a, input_b, max_element_count);

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
