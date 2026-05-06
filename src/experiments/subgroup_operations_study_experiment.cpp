#include "experiments/subgroup_operations_study_experiment.hpp"

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

constexpr const char* kExperimentId = "41_subgroup_operations_study";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTargetElementCount = 134217728U;
constexpr uint32_t kMinimumElementCount = kWorkgroupSize;

enum class VariantKind : uint32_t {
    SharedBaseline = 0U,
    SubgroupIntrinsics = 1U,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* name;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::SharedBaseline, "shared_baseline"},
    {VariantKind::SubgroupIntrinsics, "subgroup_intrinsics"},
}};

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
    uint32_t element_count = 0U;
    uint32_t variant_mode = 0U;
    uint32_t reserved0 = 0U;
    uint32_t reserved1 = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct SubgroupSupportInfo {
    uint32_t subgroup_size = 0U;
    bool compute_stage_supported = false;
    bool arithmetic_supported = false;
    bool ballot_supported = false;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    uint32_t value = index * 747796405U + 2891336453U;
    value ^= value >> 16U;
    value *= 2246822519U;
    value ^= value >> 13U;
    value *= 3266489917U;
    value ^= value >> 16U;
    return value;
}

void fill_input_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = input_pattern_value(index);
    }
}

bool validate_input_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != input_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

SubgroupSupportInfo query_subgroup_support(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties);

    return SubgroupSupportInfo{
        .subgroup_size = subgroup_properties.subgroupSize,
        .compute_stage_supported = (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0U,
        .arithmetic_supported = (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0U,
        .ballot_supported = (subgroup_properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0U,
    };
}

std::vector<uint32_t> build_reference_outputs(const uint32_t* input_values, uint32_t element_count,
                                              uint32_t subgroup_size) {
    std::vector<uint32_t> expected(element_count, 0U);
    if (subgroup_size == 0U) {
        return expected;
    }

    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    for (uint32_t group = 0U; group < group_count_x; ++group) {
        const uint32_t group_base = group * kWorkgroupSize;
        for (uint32_t subgroup_base = 0U; subgroup_base < kWorkgroupSize; subgroup_base += subgroup_size) {
            const uint32_t subgroup_end = std::min(subgroup_base + subgroup_size, kWorkgroupSize);
            uint32_t subgroup_sum = 0U;
            uint32_t active_count = 0U;

            for (uint32_t lane = subgroup_base; lane < subgroup_end; ++lane) {
                const uint32_t global_index = group_base + lane;
                const uint32_t value = global_index < element_count ? input_values[global_index] : 0U;
                subgroup_sum += value;
                active_count += (value & 1U) == 0U ? 0U : 1U;
            }

            const uint32_t subgroup_result = subgroup_sum + (active_count * 17U);
            for (uint32_t lane = subgroup_base; lane < subgroup_end; ++lane) {
                const uint32_t global_index = group_base + lane;
                if (global_index < element_count) {
                    expected[global_index] = subgroup_result;
                }
            }
        }
    }

    return expected;
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
                                out_resources.input_buffer)) {
        std::cerr << "Failed to create subgroup-operations input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_element_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create subgroup-operations output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "subgroup-operations input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "subgroup-operations output buffer",
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

    // Shader binding contract:
    //   0 -> identical input stream for both subgroup variants
    //   1 -> per-invocation aggregate results for correctness comparison
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load subgroup-operations shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create subgroup-operations descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create subgroup-operations descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate subgroup-operations descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create subgroup-operations pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create subgroup-operations compute pipeline.\n";
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
                    VariantKind variant_kind, uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // variant_mode selects the shader path:
    //   0 = shared-memory baseline
    //   1 = subgroup intrinsic implementation
    const PushConstants push_constants{
        .element_count = element_count,
        .variant_mode = static_cast<uint32_t>(variant_kind),
        .reserved0 = 0U,
        .reserved1 = 0U,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& expected,
                            uint32_t element_count) {
    if (output_values == nullptr || expected.size() < element_count) {
        return false;
    }
    return std::equal(expected.begin(), expected.begin() + element_count, output_values);
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       uint32_t subgroup_size, uint64_t logical_payload_bytes, bool correctness_pass,
                       bool dispatch_ok) {
    append_note(notes, "variant=" + std::string(descriptor.name));
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "subgroup_size=" + std::to_string(subgroup_size));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
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
              uint32_t subgroup_size, uint32_t max_dispatch_groups_x, SubgroupOperationsStudyExperimentOutput& output,
              bool verbose_progress) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << descriptor.name << ".\n";
        return false;
    }

    const std::vector<uint32_t> reference_outputs = build_reference_outputs(input_values, element_count, subgroup_size);
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(element_count);
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));
    const bool input_valid = validate_input_values(input_values, element_count);

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && input_valid && validate_output_values(output_values, reference_outputs, element_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, element_count, descriptor.kind, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid && validate_output_values(output_values, reference_outputs, element_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, element_count, subgroup_size, logical_payload_bytes, correctness_pass,
                          dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.name,
            .problem_size = element_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(logical_payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

} // namespace

SubgroupOperationsStudyExperimentOutput
run_subgroup_operations_study_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                         const SubgroupOperationsStudyExperimentConfig& config) {
    SubgroupOperationsStudyExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "subgroup operations study experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const SubgroupSupportInfo subgroup_support = query_subgroup_support(context.physical_device());
    if (!subgroup_support.compute_stage_supported || !subgroup_support.arithmetic_supported ||
        !subgroup_support.ballot_supported || subgroup_support.subgroup_size == 0U) {
        std::cerr << "Selected GPU does not support required compute-stage subgroup arithmetic+ballot features.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "41_subgroup_operations_study.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for subgroup operations study experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for subgroup operations study experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_element_count
                  << ", input_span_bytes=" << compute_span_bytes(max_element_count)
                  << ", output_span_bytes=" << compute_span_bytes(max_element_count)
                  << ", subgroup_size=" << subgroup_support.subgroup_size
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
            if (!run_case(context, runner, pipeline, buffers, descriptor, element_count, subgroup_support.subgroup_size,
                          max_dispatch_groups_x, output, config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
