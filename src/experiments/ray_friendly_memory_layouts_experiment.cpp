#include "experiments/ray_friendly_memory_layouts_experiment.hpp"

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

constexpr const char* kExperimentId = "43_ray_friendly_memory_layouts";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTargetPrimitiveCount = 8388608U;
constexpr uint32_t kMinimumPrimitiveCount = kWorkgroupSize;
constexpr uint32_t kAoSStrideWords = 16U;
constexpr uint32_t kActiveWords = 8U;
constexpr uint32_t kMappingSeed = 0x9E3779B9U;

enum class LayoutMode : uint32_t {
    AoS64 = 0U,
    SoA32 = 1U,
};

enum class AccessMode : uint32_t {
    Sequential = 0U,
    Hashed = 1U,
};

struct VariantDescriptor {
    LayoutMode layout_mode;
    AccessMode access_mode;
    const char* variant_name;
};

constexpr std::array<VariantDescriptor, 4> kVariantDescriptors = {{
    {LayoutMode::AoS64, AccessMode::Sequential, "aos64_sequential"},
    {LayoutMode::SoA32, AccessMode::Sequential, "soa32_sequential"},
    {LayoutMode::AoS64, AccessMode::Hashed, "aos64_hashed"},
    {LayoutMode::SoA32, AccessMode::Hashed, "soa32_hashed"},
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
    uint32_t primitive_count = 0U;
    uint32_t layout_mode = 0U;
    uint32_t access_mode = 0U;
    uint32_t aos_stride_words = 0U;
    uint32_t active_words = 0U;
    uint32_t aos_base_words = 0U;
    uint32_t soa_base_words = 0U;
    uint32_t soa_stride_words = 0U;
    uint32_t mapping_seed = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 9U));

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

uint32_t source_pattern(uint32_t primitive_index, uint32_t word_index) {
    return hash32((primitive_index * 131U) ^ (word_index * 17U) ^ 0x6A09E667U);
}

uint32_t resolve_source_index(uint32_t index, uint32_t primitive_count, AccessMode access_mode) {
    if (access_mode == AccessMode::Hashed) {
        return hash32(index ^ kMappingSeed) % primitive_count;
    }
    return index;
}

uint32_t expected_output_value(const uint32_t* source_words, uint32_t primitive_index, uint32_t primitive_count,
                               const VariantDescriptor& variant, uint32_t soa_base_words, uint32_t soa_stride_words) {
    const uint32_t source_index = resolve_source_index(primitive_index, primitive_count, variant.access_mode);
    uint32_t state = 0x811C9DC5U ^ (primitive_index * 0x9E3779B9U);

    for (uint32_t word = 0U; word < kActiveWords; ++word) {
        uint32_t value = 0U;
        if (variant.layout_mode == LayoutMode::AoS64) {
            const uint32_t offset = source_index * kAoSStrideWords + word;
            value = source_words[offset];
        } else {
            const uint32_t offset = soa_base_words + (word * soa_stride_words) + source_index;
            value = source_words[offset];
        }
        state ^= value + (word * 0x85EBCA6BU);
        state = (state << 5U) | (state >> 27U);
        state *= 0xC2B2AE35U;
    }
    return state;
}

uint32_t determine_primitive_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t source_words_per_primitive = kAoSStrideWords + kActiveWords;
    const uint64_t source_capacity =
        static_cast<uint64_t>(max_buffer_bytes) / (source_words_per_primitive * sizeof(uint32_t));
    const uint64_t output_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped =
        std::min({source_capacity, output_capacity, dispatch_capacity, static_cast<uint64_t>(kTargetPrimitiveCount)});
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kMinimumPrimitiveCount) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_primitive_count) {
    const std::array<uint32_t, 5> base_sizes = {262144U, 1048576U, 2097152U, 4194304U, 8388608U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_primitive_count) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_primitive_count) {
        sizes.push_back(max_primitive_count);
    }
    return sizes;
}

VkDeviceSize compute_source_span_bytes(uint32_t max_primitive_count) {
    const uint64_t words = static_cast<uint64_t>(max_primitive_count) * (kAoSStrideWords + kActiveWords);
    return static_cast<VkDeviceSize>(words * sizeof(uint32_t));
}

VkDeviceSize compute_output_span_bytes(uint32_t primitive_count) {
    return static_cast<VkDeviceSize>(primitive_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t compute_soa_base_words(uint32_t max_primitive_count) {
    return max_primitive_count * kAoSStrideWords;
}

uint64_t compute_logical_payload_bytes(uint32_t primitive_count) {
    return static_cast<uint64_t>(primitive_count) * static_cast<uint64_t>((kActiveWords + 1U) * sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_primitive_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(),
                                compute_source_span_bytes(max_primitive_count), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.source_buffer)) {
        std::cerr << "Failed to create ray-layout source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(),
                                compute_output_span_bytes(max_primitive_count), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create ray-layout output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.source_buffer, "ray-layout source buffer",
                           out_resources.source_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "ray-layout output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.source_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.source_buffer.memory);
            out_resources.source_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
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
    //   0 -> shared backing storage interpreted as AoS64 or SoA32
    //   1 -> one traversal-style checksum per logical primitive
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, source_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load ray-layout shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create ray-layout descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create ray-layout descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate ray-layout descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create ray-layout pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create ray-layout compute pipeline.\n";
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

void fill_source_values(uint32_t* source_words, uint32_t max_primitive_count) {
    const uint32_t soa_base_words = compute_soa_base_words(max_primitive_count);
    for (uint32_t primitive = 0U; primitive < max_primitive_count; ++primitive) {
        for (uint32_t word = 0U; word < kAoSStrideWords; ++word) {
            uint32_t value = source_pattern(primitive, word + 53U);
            if (word < kActiveWords) {
                value = source_pattern(primitive, word);
            }
            source_words[(primitive * kAoSStrideWords) + word] = value;
        }
        for (uint32_t word = 0U; word < kActiveWords; ++word) {
            source_words[soa_base_words + (word * max_primitive_count) + primitive] = source_pattern(primitive, word);
        }
    }
}

bool validate_source_values(const uint32_t* source_words, uint32_t max_primitive_count) {
    const uint32_t soa_base_words = compute_soa_base_words(max_primitive_count);
    for (uint32_t primitive = 0U; primitive < max_primitive_count; ++primitive) {
        for (uint32_t word = 0U; word < kActiveWords; ++word) {
            const uint32_t aos_value = source_words[(primitive * kAoSStrideWords) + word];
            const uint32_t soa_value = source_words[soa_base_words + (word * max_primitive_count) + primitive];
            if (aos_value != source_pattern(primitive, word) || soa_value != source_pattern(primitive, word)) {
                return false;
            }
        }
    }
    return true;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, const PushConstants& constants,
                    uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x =
        VulkanComputeUtils::compute_group_count_1d(constants.primitive_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

bool validate_output_values(const uint32_t* output_words, const std::vector<uint32_t>& reference,
                            uint32_t primitive_count) {
    if (output_words == nullptr || reference.size() < primitive_count) {
        return false;
    }
    return std::equal(reference.begin(), reference.begin() + primitive_count, output_words);
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t primitive_count,
                       uint64_t logical_payload_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "variant=" + std::string(descriptor.variant_name));
    append_note(notes, "primitive_count=" + std::to_string(primitive_count));
    append_note(notes, "aos_stride_bytes=64");
    append_note(notes, "active_words=8");
    append_note(notes,
                "access_mode=" + std::string(descriptor.access_mode == AccessMode::Hashed ? "hashed" : "sequential"));
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
              const BufferResources& buffers, const VariantDescriptor& descriptor, uint32_t primitive_count,
              uint32_t max_primitive_count, uint32_t max_dispatch_groups_x,
              RayFriendlyMemoryLayoutsExperimentOutput& output, bool verbose_progress) {
    const auto* source_words = static_cast<const uint32_t*>(buffers.source_mapped_ptr);
    auto* output_words = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (source_words == nullptr || output_words == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << descriptor.variant_name
                  << ".\n";
        return false;
    }

    const bool source_valid = validate_source_values(source_words, max_primitive_count);
    const uint32_t soa_base_words = compute_soa_base_words(max_primitive_count);
    std::vector<uint32_t> reference(primitive_count, 0U);
    for (uint32_t index = 0U; index < primitive_count; ++index) {
        reference[index] = expected_output_value(source_words, index, primitive_count, descriptor, soa_base_words,
                                                 max_primitive_count);
    }

    // layout_mode selects how binding 0 is decoded, while access_mode selects
    // the primitive-to-source mapping. The base/stride words describe both
    // layouts explicitly so the shader never relies on implicit packing.
    const PushConstants constants{
        .primitive_count = primitive_count,
        .layout_mode = static_cast<uint32_t>(descriptor.layout_mode),
        .access_mode = static_cast<uint32_t>(descriptor.access_mode),
        .aos_stride_words = kAoSStrideWords,
        .active_words = kActiveWords,
        .aos_base_words = 0U,
        .soa_base_words = soa_base_words,
        .soa_stride_words = max_primitive_count,
        .mapping_seed = kMappingSeed,
    };
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(primitive_count);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        std::fill_n(output_words, primitive_count, 0U);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && source_valid && validate_output_values(output_words, reference, primitive_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", primitive_count=" << primitive_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        std::fill_n(output_words, primitive_count, 0U);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && source_valid && validate_output_values(output_words, reference, primitive_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, primitive_count, logical_payload_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", primitive_count=" << primitive_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = primitive_count,
            .dispatch_count = 1U,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(primitive_count, 1U, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(logical_payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_primitives_" + std::to_string(primitive_count),
        dispatch_samples));
    return true;
}

} // namespace

RayFriendlyMemoryLayoutsExperimentOutput
run_ray_friendly_memory_layouts_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                           const RayFriendlyMemoryLayoutsExperimentConfig& config) {
    RayFriendlyMemoryLayoutsExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "ray-friendly memory layouts experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "43_ray_friendly_memory_layouts.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for ray-friendly memory layouts experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_primitive_count = determine_primitive_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_primitive_count == 0U) {
        std::cerr << "Scratch buffer too small for ray-friendly memory layouts experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_primitives=" << max_primitive_count
                  << ", source_span_bytes=" << compute_source_span_bytes(max_primitive_count)
                  << ", output_span_bytes=" << compute_output_span_bytes(max_primitive_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_primitive_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* source_words = static_cast<uint32_t*>(buffers.source_mapped_ptr);
    if (source_words == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped source pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_source_values(source_words, max_primitive_count);

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_primitive_count);
    for (uint32_t primitive_count : problem_sizes) {
        for (const auto& descriptor : kVariantDescriptors) {
            if (!run_case(context, runner, pipeline, buffers, descriptor, primitive_count, max_primitive_count,
                          max_dispatch_groups_x, output, config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
