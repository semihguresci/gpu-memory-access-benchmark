#include "experiments/subgroup_scan_variants_experiment.hpp"

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

constexpr const char* kExperimentId = "31_subgroup_scan_variants";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTargetLogicalCount = 262144U;
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;
constexpr std::array<uint32_t, 3> kItemsPerThreadValues = {1U, 4U, 8U};

enum class VariantKind : uint32_t {
    SharedBlockScan,
    SubgroupBlockScan,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    const char* shader_filename;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {VariantKind::SharedBlockScan, "shared_block_scan", "31_subgroup_scan_shared.comp.spv"},
    {VariantKind::SubgroupBlockScan, "subgroup_block_scan", "31_subgroup_scan_subgroup.comp.spv"},
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
    uint32_t logical_count = 0U;
    uint32_t items_per_thread = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 2U));

struct SubgroupSupportInfo {
    uint32_t subgroup_size = 0U;
    bool compute_stage_supported = false;
    bool arithmetic_supported = false;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
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
    };
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus;
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

uint32_t compute_block_elements(uint32_t items_per_thread) {
    return kWorkgroupSize * items_per_thread;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint32_t largest_block_elements = compute_block_elements(kItemsPerThreadValues.back());
    const uint64_t max_from_buffer = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t max_from_dispatch = static_cast<uint64_t>(max_dispatch_groups_x) * largest_block_elements;
    const uint64_t effective_count =
        std::min({max_from_buffer, max_from_dispatch, static_cast<uint64_t>(kTargetLogicalCount)});
    if (effective_count < largest_block_elements) {
        return 0U;
    }

    const uint64_t rounded = effective_count - (effective_count % largest_block_elements);
    return rounded >= largest_block_elements ? static_cast<uint32_t>(rounded) : 0U;
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create subgroup scan input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create subgroup scan output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "subgroup scan input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "subgroup scan output buffer",
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
        std::cerr << "Failed to load subgroup scan shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create subgroup scan descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create subgroup scan descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate subgroup scan descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create subgroup scan pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create subgroup scan compute pipeline.\n";
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

std::vector<uint32_t> build_reference_scan(const uint32_t* input_values, uint32_t logical_count,
                                           uint32_t items_per_thread) {
    std::vector<uint32_t> reference(logical_count, 0U);
    const uint32_t block_elements = compute_block_elements(items_per_thread);
    const uint32_t block_count = logical_count / block_elements;

    for (uint32_t block_index = 0U; block_index < block_count; ++block_index) {
        uint32_t running_sum = 0U;
        const uint32_t block_base = block_index * block_elements;
        for (uint32_t offset = 0U; offset < block_elements; ++offset) {
            const uint32_t global_index = block_base + offset;
            running_sum += input_values[global_index];
            reference[global_index] = running_sum;
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

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t logical_count,
                    uint32_t items_per_thread) {
    const uint32_t block_elements = compute_block_elements(items_per_thread);
    const uint32_t group_count_x = logical_count / block_elements;
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{logical_count, items_per_thread};
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
                       uint32_t items_per_thread, uint32_t subgroup_size, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("scan_strategy=") + descriptor.variant_name);
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "items_per_thread=" + std::to_string(items_per_thread));
    append_note(notes, "block_elements=" + std::to_string(compute_block_elements(items_per_thread)));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "payload_bytes_per_element=8");
    if (descriptor.kind == VariantKind::SubgroupBlockScan) {
        append_note(notes, "subgroup_size=" + std::to_string(subgroup_size));
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
              uint32_t items_per_thread, uint32_t subgroup_size, SubgroupScanVariantsExperimentOutput& output,
              bool verbose_progress) {
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped input or output pointer for variant=" << descriptor.variant_name << ".\n";
        return false;
    }

    const std::vector<uint32_t> reference_values = build_reference_scan(input_values, logical_count, items_per_thread);
    const uint64_t payload_bytes = static_cast<uint64_t>(logical_count) * sizeof(uint32_t) * 2U;
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, items_per_thread);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, logical_count) &&
                             validate_output_values(output_values, reference_values);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", items_per_thread=" << items_per_thread
                      << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, items_per_thread);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass = dispatch_ok && validate_input_values(input_values, logical_count) &&
                                      validate_output_values(output_values, reference_values);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, logical_count, items_per_thread, subgroup_size, correctness_pass,
                          dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = std::string(descriptor.variant_name) + "_items_" + std::to_string(items_per_thread),
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
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_items_" + std::to_string(items_per_thread) +
            "_elements_" + std::to_string(logical_count),
        dispatch_samples));
    return true;
}

} // namespace

SubgroupScanVariantsExperimentOutput
run_subgroup_scan_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const SubgroupScanVariantsExperimentConfig& config) {
    SubgroupScanVariantsExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "subgroup scan variants experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const SubgroupSupportInfo subgroup_support = query_subgroup_support(context.physical_device());
    if (!subgroup_support.compute_stage_supported || !subgroup_support.arithmetic_supported ||
        subgroup_support.subgroup_size == 0U) {
        std::cerr << "Selected GPU does not support compute-stage subgroup arithmetic required for Experiment 31.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        const std::string user_path = kVariantDescriptors[index].kind == VariantKind::SharedBlockScan
                                          ? config.shared_scan_shader_path
                                          : config.subgroup_scan_shader_path;
        shader_paths[index] =
            VulkanComputeUtils::resolve_shader_path(user_path, kVariantDescriptors[index].shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for subgroup scan variant "
                      << kVariantDescriptors[index].variant_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for subgroup scan variants experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_span_bytes(logical_count), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped input or output buffer pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_input_values(input_values, logical_count);
    fill_output_values(output_values, logical_count);

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

    for (const uint32_t items_per_thread : kItemsPerThreadValues) {
        for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
            if (!run_case(context, runner, buffers, pipeline_resources[index], kVariantDescriptors[index],
                          logical_count, items_per_thread, subgroup_support.subgroup_size, output,
                          config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);
    return output;
}
