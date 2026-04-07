#include "experiments/branch_divergence_experiment.hpp"

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

constexpr const char* kExperimentId = "19_branch_divergence";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kBranchWorkRounds = 6U;
constexpr uint32_t kInputSeed = 0x12345678U;
constexpr uint32_t kPredicateSeed = 0x6D2B79F5U;
constexpr uint32_t kTrueSalt = 0xA5A5A5A5U;
constexpr uint32_t kFalseSalt = 0x3C3C3C3CU;
constexpr uint32_t kHashMultiplierA = 0x7FEB352DU;
constexpr uint32_t kHashMultiplierB = 0x846CA68BU;
constexpr uint32_t kMixMultiplier = 0x9E3779B9U;
constexpr uint32_t kMixOffset = 0x7F4A7C15U;
constexpr uint32_t kFinalMultiplier = 0x85EBCA6BU;
constexpr uint32_t kFinalOffset = 0xC2B2AE35U;

enum class PatternMode : uint32_t {
    UniformTrue = 0U,
    UniformFalse = 1U,
    Alternating = 2U,
    Random25 = 3U,
    Random50 = 4U,
    Random75 = 5U,
};

struct PatternDescriptor {
    PatternMode mode;
    const char* variant_name;
    double true_probability;
    uint32_t random_threshold;
};

struct CountResolution {
    uint32_t logical_count = 0U;
    bool rounded_to_workgroup_multiple = false;
};

constexpr std::array<PatternDescriptor, 6> kPatternDescriptors = {{
    {PatternMode::UniformTrue, "uniform_true", 1.0, 0xFFFFFFFFU},
    {PatternMode::UniformFalse, "uniform_false", 0.0, 0U},
    {PatternMode::Alternating, "alternating", 0.5, 0x80000000U},
    {PatternMode::Random25, "random_p25", 0.25, 0x40000000U},
    {PatternMode::Random50, "random_p50", 0.5, 0x80000000U},
    {PatternMode::Random75, "random_p75", 0.75, 0xC0000000U},
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
    uint32_t pattern_mode = 0U;
    uint32_t random_threshold = 0U;
    uint32_t random_seed = 0U;
    uint32_t branch_work_rounds = 0U;
    uint32_t true_salt = 0U;
    uint32_t false_salt = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 7U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

constexpr uint32_t rotl32(uint32_t value, uint32_t shift) {
    return (value << shift) | (value >> (32U - shift));
}

uint32_t source_pattern_value(uint32_t index) {
    uint32_t value = index ^ kInputSeed;
    value ^= value >> 16U;
    value *= kHashMultiplierA;
    value ^= value >> 15U;
    value *= kHashMultiplierB;
    value ^= value >> 16U;
    return value;
}

uint32_t hash32(uint32_t value) {
    value ^= value >> 16U;
    value *= kHashMultiplierA;
    value ^= value >> 15U;
    value *= kHashMultiplierB;
    value ^= value >> 16U;
    return value;
}

bool should_take_true(PatternMode mode, uint32_t index, uint32_t random_threshold) {
    switch (mode) {
    case PatternMode::UniformTrue:
        return true;
    case PatternMode::UniformFalse:
        return false;
    case PatternMode::Alternating:
        return (index & 1U) == 0U;
    case PatternMode::Random25:
    case PatternMode::Random50:
    case PatternMode::Random75:
        return hash32(index ^ kPredicateSeed) < random_threshold;
    }

    return false;
}

uint32_t branch_mix(uint32_t state, uint32_t salt, uint32_t branch_work_rounds) {
    for (uint32_t round = 0U; round < branch_work_rounds; ++round) {
        const uint32_t round_salt = salt ^ (round * kMixMultiplier);
        state += round_salt + kMixOffset;
        state ^= state >> 16U;
        state = rotl32(state, 7U);
        state *= kFinalMultiplier;
        state ^= state >> 13U;
        state += kFinalOffset;
    }

    return state;
}

uint32_t expected_output_value(uint32_t index, PatternMode mode, uint32_t random_threshold) {
    const uint32_t input_value = source_pattern_value(index);
    const uint32_t seed = input_value ^ (index * kMixMultiplier) ^ kPredicateSeed;
    const uint32_t branch_salt = should_take_true(mode, index, random_threshold) ? kTrueSalt : kFalseSalt;
    return branch_mix(seed, branch_salt, kBranchWorkRounds);
}

void fill_source_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_destination_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, 0U);
}

bool validate_source_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != source_pattern_value(index)) {
            return false;
        }
    }

    return true;
}

bool validate_destination_values(const uint32_t* values, uint32_t element_count, PatternMode mode,
                                 uint32_t random_threshold) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != expected_output_value(index, mode, random_threshold)) {
            return false;
        }
    }

    return true;
}

CountResolution determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    CountResolution resolution{};
    const uint64_t buffer_capacity_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity_elements = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_elements = std::min({buffer_capacity_elements, dispatch_capacity_elements,
                                                  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_elements < kWorkgroupSize) {
        return resolution;
    }

    const uint64_t rounded_elements = effective_elements - (effective_elements % kWorkgroupSize);
    resolution.logical_count = static_cast<uint32_t>(rounded_elements);
    resolution.rounded_to_workgroup_multiple = (rounded_elements != effective_elements);
    return resolution;
}

VkDeviceSize compute_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize buffer_size, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.src_buffer)) {
        std::cerr << "Failed to create branch divergence source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create branch divergence destination buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "branch divergence source buffer",
                           out_resources.src_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "branch divergence destination buffer",
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

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load branch divergence shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create branch divergence descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create branch divergence descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate branch divergence descriptor set.\n";
        return false;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create branch divergence pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create branch divergence compute pipeline.\n";
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
                    PatternMode mode, uint32_t random_threshold) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        logical_count, static_cast<uint32_t>(mode), random_threshold, kPredicateSeed, kBranchWorkRounds, kTrueSalt,
        kFalseSalt,
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

const char* pattern_mode_name(PatternMode mode) {
    switch (mode) {
    case PatternMode::UniformTrue:
        return "uniform_true";
    case PatternMode::UniformFalse:
        return "uniform_false";
    case PatternMode::Alternating:
        return "alternating";
    case PatternMode::Random25:
        return "random_p25";
    case PatternMode::Random50:
        return "random_p50";
    case PatternMode::Random75:
        return "random_p75";
    }

    return "unknown";
}

void record_case_notes(std::string& notes, PatternMode mode, uint32_t logical_count, uint32_t group_count_x,
                       uint32_t random_threshold, bool rounded_to_workgroup_multiple, bool correctness_pass,
                       bool dispatch_ok, std::size_t scratch_size_bytes) {
    append_note(notes, std::string("pattern_mode=") + pattern_mode_name(mode));
    append_note(notes, "expected_true_probability=" + std::to_string((mode == PatternMode::UniformTrue)    ? 1.0
                                                                     : (mode == PatternMode::UniformFalse) ? 0.0
                                                                     : (mode == PatternMode::Alternating)  ? 0.5
                                                                     : (mode == PatternMode::Random25)     ? 0.25
                                                                     : (mode == PatternMode::Random50)     ? 0.5
                                                                                                           : 0.75));
    append_note(notes, "random_threshold=" + std::to_string(random_threshold));
    append_note(notes, "random_seed=" + std::to_string(kPredicateSeed));
    append_note(notes, "branch_work_rounds=" + std::to_string(kBranchWorkRounds));
    append_note(notes, "true_salt=" + std::to_string(kTrueSalt));
    append_note(notes, "false_salt=" + std::to_string(kFalseSalt));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "source_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_span_bytes(logical_count))));
    append_note(notes, "destination_span_bytes=" +
                           std::to_string(static_cast<unsigned long long>(compute_span_bytes(logical_count))));
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    append_note(notes, "bytes_per_element=" + std::to_string(sizeof(uint32_t) * 2U));
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

bool run_pattern_case(VulkanContext& context, const BenchmarkRunner& runner,
                      const PipelineResources& pipeline_resources, const BufferResources& buffers,
                      PatternDescriptor descriptor, uint32_t logical_count, bool rounded_to_workgroup_multiple,
                      BranchDivergenceExperimentOutput& output, bool verbose_progress, std::size_t scratch_size_bytes,
                      std::size_t per_buffer_budget_bytes) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for " << descriptor.variant_name
                  << ".\n";
        return false;
    }

    const auto* src_values = static_cast<const uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for variant " << descriptor.variant_name
                  << ".\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << descriptor.variant_name
                  << ", logical_elements=" << logical_count << ", group_count_x=" << group_count_x
                  << ", scratch_size_bytes=" << scratch_size_bytes
                  << ", per_buffer_budget_bytes=" << per_buffer_budget_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, logical_count, descriptor.mode, descriptor.random_threshold);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_source_values(src_values, logical_count) &&
            validate_destination_values(dst_values, logical_count, descriptor.mode, descriptor.random_threshold);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << descriptor.variant_name
                      << ", logical_elements=" << logical_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    const VkDeviceSize total_bytes = compute_span_bytes(logical_count) * 2U;
    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_destination_values(dst_values, logical_count);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, logical_count, descriptor.mode, descriptor.random_threshold);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_source_values(src_values, logical_count) &&
            validate_destination_values(dst_values, logical_count, descriptor.mode, descriptor.random_threshold);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor.mode, logical_count, group_count_x, descriptor.random_threshold,
                          rounded_to_workgroup_multiple, correctness_pass, dispatch_ok, scratch_size_bytes);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
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
        std::string(kExperimentId) + "_" + descriptor.variant_name, dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << descriptor.variant_name
                  << ", logical_elements=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

BranchDivergenceExperimentOutput run_branch_divergence_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                                  const BranchDivergenceExperimentConfig& config) {
    BranchDivergenceExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "branch divergence experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "19_branch_divergence.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for branch divergence experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const CountResolution count_resolution =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (count_resolution.logical_count == 0U) {
        std::cerr << "Scratch buffer too small for branch divergence experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t logical_count = count_resolution.logical_count;
    const VkDeviceSize buffer_span_bytes = compute_span_bytes(logical_count);
    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_outputs=" << logical_count
                  << ", buffer_span_bytes=" << buffer_span_bytes
                  << ", per_buffer_budget_bytes=" << config.max_buffer_bytes
                  << ", scratch_size_bytes=" << config.scratch_size_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, buffer_span_bytes, buffers)) {
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

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    update_descriptor_set(context, buffers, pipeline_resources.descriptor_set);

    for (const PatternDescriptor& descriptor : kPatternDescriptors) {
        if (!run_pattern_case(context, runner, pipeline_resources, buffers, descriptor, logical_count,
                              count_resolution.rounded_to_workgroup_multiple, output, config.verbose_progress,
                              config.scratch_size_bytes, config.max_buffer_bytes)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);
    destroy_buffer_resources(context, buffers);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
