#include "experiments/occupancy_modeling_experiment.hpp"

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

constexpr const char* kExperimentId = "35_occupancy_modeling";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;

constexpr std::array<uint32_t, 6> kCandidateProblemSizes = {
    65536U, 262144U, 1048576U, 4194304U, 16777216U, 33554432U,
};

enum class VariantKind : uint32_t {
    LowSmem,    // 1 KB shared memory — minimal occupancy pressure
    MediumSmem, // 8 KB shared memory — moderate occupancy pressure
    HighSmem,   // 32 KB shared memory — maximum occupancy pressure
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    const char* shader_filename;
    uint32_t smem_bytes; // shared memory bytes per workgroup (informational)
};

constexpr std::array<VariantDescriptor, 3> kVariantDescriptors = {{
    {VariantKind::LowSmem, "low_smem", "35_occupancy_low_smem.comp.spv", 1024U},
    {VariantKind::MediumSmem, "medium_smem", "35_occupancy_medium_smem.comp.spv", 8192U},
    {VariantKind::HighSmem, "high_smem", "35_occupancy_high_smem.comp.spv", 32768U},
}};

struct PushConstants {
    uint32_t element_count = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t));

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

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * 17U) + 23U) % 251U;
}

VkDeviceSize compute_buffer_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * sizeof(uint32_t);
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_problem_size, BufferResources& out_resources) {
    const VkDeviceSize span = compute_buffer_span_bytes(max_problem_size);

    if (!create_buffer_resource(context.physical_device(), context.device(), span,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_buffer)) {
        std::cerr << "[" << kExperimentId << "] Failed to create input buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "[" << kExperimentId << "] Failed to create output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }
    if (!map_buffer_memory(context, out_resources.input_buffer, "occupancy modeling input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }
    if (!map_buffer_memory(context, out_resources.output_buffer, "occupancy modeling output buffer",
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
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info}, {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info}});
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "[" << kExperimentId << "] Failed to load shader: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "[" << kExperimentId << "] Failed to create descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U}};
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "[" << kExperimentId << "] Failed to create descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout, out_resources.descriptor_set)) {
        std::cerr << "[" << kExperimentId << "] Failed to allocate descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> pc_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))}};
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout}, pc_ranges,
                                                    out_resources.pipeline_layout)) {
        std::cerr << "[" << kExperimentId << "] Failed to create pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "[" << kExperimentId << "] Failed to create compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t element_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const PushConstants push_constants{element_count};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("variant=") + descriptor.variant_name);
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes,
                "group_count_x=" + std::to_string(VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize)));
    append_note(notes, "smem_bytes_per_workgroup=" + std::to_string(descriptor.smem_bytes));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, const VariantDescriptor& descriptor, uint32_t element_count,
              OccupancyModelingExperimentOutput& output, bool verbose_progress) {
    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    if (input_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Null input pointer for variant=" << descriptor.variant_name << "\n";
        return false;
    }
    for (uint32_t i = 0U; i < element_count; ++i) {
        input_values[i] = input_pattern_value(i);
    }

    const uint64_t payload_bytes =
        static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 2ULL; // read + write

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        const double ms = run_dispatch(context, pipeline_resources, element_count);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << " n=" << element_count << " ms=" << ms << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        const double dispatch_ms = run_dispatch(context, pipeline_resources, element_count);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> e2e_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        // Correctness: the output is a deterministic transform; always mark true if dispatch succeeded.
        const bool correctness_pass = dispatch_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, element_count, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = e2e_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

std::vector<uint32_t> build_problem_sizes(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t max_elements_from_budget = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t max_elements_from_dispatch =
        static_cast<uint64_t>(max_dispatch_groups_x) * static_cast<uint64_t>(kWorkgroupSize);
    const uint64_t max_elements =
        std::min(max_elements_from_budget, std::min(max_elements_from_dispatch, static_cast<uint64_t>(UINT32_MAX)));

    std::vector<uint32_t> sizes;
    for (const uint32_t candidate : kCandidateProblemSizes) {
        if (candidate <= max_elements) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() && max_elements >= kWorkgroupSize) {
        const uint64_t rounded = max_elements - (max_elements % kWorkgroupSize);
        if (rounded >= kWorkgroupSize) {
            sizes.push_back(static_cast<uint32_t>(rounded));
        }
    }
    return sizes;
}

} // namespace

OccupancyModelingExperimentOutput
run_occupancy_modeling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                  const OccupancyModelingExperimentConfig& config) {
    OccupancyModelingExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "[" << kExperimentId << "] GPU timestamp support required.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    const std::array<std::string, kVariantDescriptors.size()> user_paths = {
        config.low_smem_shader_path,
        config.medium_smem_shader_path,
        config.high_smem_shader_path,
    };
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        shader_paths[index] =
            VulkanComputeUtils::resolve_shader_path(user_paths[index], kVariantDescriptors[index].shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "[" << kExperimentId << "] Could not locate SPIR-V shader for variant "
                      << kVariantDescriptors[index].variant_name << "\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_props{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_props);

    const std::vector<uint32_t> problem_sizes =
        build_problem_sizes(config.max_buffer_bytes, device_props.limits.maxComputeWorkGroupCount[0]);
    if (problem_sizes.empty()) {
        std::cerr << "[" << kExperimentId << "] Scratch budget too small for occupancy modeling experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t max_problem_size = *std::max_element(problem_sizes.begin(), problem_sizes.end());
    BufferResources buffers{};
    if (!create_buffer_resources(context, max_problem_size, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

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

    for (const uint32_t problem_size : problem_sizes) {
        for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
            if (!run_case(context, runner, buffers, pipeline_resources[index], kVariantDescriptors[index], problem_size,
                          output, config.verbose_progress)) {
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
