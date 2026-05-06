#include "experiments/bvh_node_layout_experiment.hpp"

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

constexpr const char* kExperimentId = "36_bvh_node_layout";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kMaxWordsPerNode = 16U;
constexpr uint32_t kActiveWordsPerNode = 8U;
constexpr uint32_t kCompactWordsPerNode = 8U;
constexpr uint32_t kOutputWordsPerNode = 1U;
constexpr uint32_t kTargetNodeCount = 8388608U;
constexpr uint32_t kNodePatternSeed = 0x6A09E667U;
constexpr uint32_t kHashMul0 = 0x7FEB352DU;
constexpr uint32_t kHashMul1 = 0x846CA68BU;
constexpr uint32_t kOutputSentinel = 0xDEADBEEFU;
constexpr uint32_t kHashedAccessSeed = 0x9E3779B9U;

enum class LayoutMode : uint32_t {
    CompactSequential = 0U,
    PaddedSequential = 1U,
    CompactHashed = 2U,
};

struct VariantDescriptor {
    LayoutMode mode;
    const char* variant_name;
    uint32_t words_per_node;
    bool use_padded_region;
    uint32_t storage_bytes_per_node;
    const char* access_pattern;
};

constexpr std::array<VariantDescriptor, 3> kVariantDescriptors = {{
    {LayoutMode::CompactSequential, "compact32_sequential", kCompactWordsPerNode, false, 32U, "sequential"},
    {LayoutMode::PaddedSequential, "padded64_sequential", kMaxWordsPerNode, true, 64U, "sequential"},
    {LayoutMode::CompactHashed, "compact32_hashed", kCompactWordsPerNode, false, 32U, "hashed"},
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
    uint32_t node_count = 0U;
    uint32_t words_per_node = 0U;
    uint32_t layout_mode = 0U;
    uint32_t hashed_access_seed = 0U;
    uint32_t base_word_offset = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 5U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t hash32(uint32_t value) {
    value ^= value >> 16U;
    value *= kHashMul0;
    value ^= value >> 15U;
    value *= kHashMul1;
    value ^= value >> 16U;
    return value;
}

uint32_t source_word_value(uint32_t node_index, uint32_t word_index) {
    return hash32((node_index * 131U) ^ (word_index * 17U) ^ kNodePatternSeed);
}

uint32_t resolve_source_node(uint32_t node_index, uint32_t node_count, LayoutMode mode) {
    if (mode == LayoutMode::CompactHashed) {
        return hash32(node_index ^ kHashedAccessSeed) % node_count;
    }
    return node_index;
}

uint32_t resolve_word_offset(LayoutMode mode, uint32_t active_word_index) {
    if (mode == LayoutMode::PaddedSequential) {
        return active_word_index * 2U;
    }
    return active_word_index;
}

uint32_t expected_output_value(const uint32_t* source_words, uint32_t node_index, uint32_t node_count,
                               const VariantDescriptor& variant, uint32_t base_word_offset) {
    const uint32_t source_node = resolve_source_node(node_index, node_count, variant.mode);
    const uint32_t base = base_word_offset + (source_node * variant.words_per_node);

    uint32_t state = 0x811C9DC5U ^ (node_index * 0x9E3779B9U);
    for (uint32_t word = 0U; word < kActiveWordsPerNode; ++word) {
        const uint32_t offset = resolve_word_offset(variant.mode, word);
        const uint32_t value = source_words[base + offset];
        state ^= value + (word * 0x85EBCA6BU);
        state = (state << 5U) | (state >> 27U);
        state *= 0xC2B2AE35U;
    }
    return state;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    constexpr uint32_t kSourceWordsPerNodeTotal = kCompactWordsPerNode + kMaxWordsPerNode;
    const uint64_t source_capacity_nodes =
        static_cast<uint64_t>(max_buffer_bytes) / (static_cast<uint64_t>(kSourceWordsPerNodeTotal) * sizeof(uint32_t));
    const uint64_t output_capacity_nodes = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity_nodes = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;

    const uint64_t capped_nodes = std::min({source_capacity_nodes, output_capacity_nodes, dispatch_capacity_nodes,
                                            static_cast<uint64_t>(kTargetNodeCount),
                                            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (capped_nodes < kWorkgroupSize) {
        return 0U;
    }

    return static_cast<uint32_t>(capped_nodes - (capped_nodes % kWorkgroupSize));
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_nodes) {
    const std::array<uint32_t, 6> base_sizes = {262144U, 524288U, 1048576U, 2097152U, 4194304U, 8388608U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_nodes) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_nodes) {
        sizes.push_back(max_nodes);
    }
    return sizes;
}

VkDeviceSize compute_source_span_bytes(uint32_t node_count) {
    return static_cast<VkDeviceSize>(node_count) * static_cast<VkDeviceSize>(kCompactWordsPerNode + kMaxWordsPerNode) *
           static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_output_span_bytes(uint32_t node_count) {
    return static_cast<VkDeviceSize>(node_count) * static_cast<VkDeviceSize>(kOutputWordsPerNode) *
           static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_nodes, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_source_span_bytes(max_nodes),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.source_buffer)) {
        std::cerr << "Failed to create BVH node layout source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_output_span_bytes(max_nodes),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create BVH node layout output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.source_buffer, "bvh node layout source buffer",
                           out_resources.source_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "bvh node layout output buffer",
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
    //   0 -> backing words for the active BVH layout region
    //   1 -> one checksum per logical node traversal
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = source_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = output_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load BVH node layout shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create BVH node layout descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create BVH node layout descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate BVH node layout descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create BVH node layout pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create BVH node layout compute pipeline.\n";
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

void fill_source_values(uint32_t* source_words, uint32_t max_nodes) {
    const uint32_t compact_region_words = max_nodes * kCompactWordsPerNode;
    for (uint32_t node = 0U; node < max_nodes; ++node) {
        const uint32_t compact_base = node * kCompactWordsPerNode;
        for (uint32_t word = 0U; word < kCompactWordsPerNode; ++word) {
            source_words[compact_base + word] = source_word_value(node, word);
        }

        const uint32_t padded_base = compact_region_words + (node * kMaxWordsPerNode);
        for (uint32_t word = 0U; word < kMaxWordsPerNode; ++word) {
            source_words[padded_base + word] = source_word_value(node, word + 97U);
        }
        for (uint32_t word = 0U; word < kActiveWordsPerNode; ++word) {
            source_words[padded_base + (word * 2U)] = source_word_value(node, word);
        }
    }
}

bool validate_source_values(const uint32_t* source_words, uint32_t total_nodes, uint32_t active_nodes) {
    const uint32_t compact_region_words = total_nodes * kCompactWordsPerNode;
    for (uint32_t node = 0U; node < active_nodes; ++node) {
        const uint32_t compact_base = node * kCompactWordsPerNode;
        for (uint32_t word = 0U; word < kCompactWordsPerNode; ++word) {
            if (source_words[compact_base + word] != source_word_value(node, word)) {
                return false;
            }
        }

        const uint32_t padded_base = compact_region_words + (node * kMaxWordsPerNode);
        for (uint32_t word = 0U; word < kMaxWordsPerNode; ++word) {
            uint32_t expected = source_word_value(node, word + 97U);
            if ((word % 2U) == 0U && (word / 2U) < kActiveWordsPerNode) {
                expected = source_word_value(node, word / 2U);
            }
            if (source_words[padded_base + word] != expected) {
                return false;
            }
        }
    }
    return true;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources,
                    const PushConstants& constants) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(constants.node_count, kWorkgroupSize);
    if (group_count_x == 0U) {
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

void record_case_notes(std::string& notes, const VariantDescriptor& variant, uint32_t node_count, bool correctness_pass,
                       bool dispatch_ok) {
    append_note(notes, "layout_mode=" + std::string(variant.variant_name));
    append_note(notes, "node_count=" + std::to_string(node_count));
    append_note(notes, "active_words_per_node=" + std::to_string(kActiveWordsPerNode));
    append_note(notes, "storage_words_per_node=" + std::to_string(variant.words_per_node));
    append_note(notes, "storage_bytes_per_node=" + std::to_string(variant.storage_bytes_per_node));
    append_note(notes, "active_bytes_per_node=" + std::to_string((kActiveWordsPerNode + kOutputWordsPerNode) * 4U));
    append_note(notes, "access_pattern=" + std::string(variant.access_pattern));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, const VariantDescriptor& variant, uint32_t node_count, uint32_t max_nodes,
              BvhNodeLayoutExperimentOutput& output, bool verbose_progress) {
    const auto* source_words = static_cast<const uint32_t*>(buffers.source_mapped_ptr);
    auto* output_words = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (source_words == nullptr || output_words == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << variant.variant_name << ".\n";
        return false;
    }

    const uint32_t base_word_offset = variant.use_padded_region ? (max_nodes * kCompactWordsPerNode) : 0U;
    std::vector<uint32_t> reference(node_count, 0U);
    for (uint32_t node = 0U; node < node_count; ++node) {
        reference[node] = expected_output_value(source_words, node, node_count, variant, base_word_offset);
    }

    // layout_mode must stay aligned with the shader-side interpretation of
    // binding 0. words_per_node/base_word_offset select the compact or padded
    // region, while hashed_access_seed is consumed only by the hashed variant.
    const PushConstants constants{
        .node_count = node_count,
        .words_per_node = variant.words_per_node,
        .layout_mode = static_cast<uint32_t>(variant.mode),
        .hashed_access_seed = kHashedAccessSeed,
        .base_word_offset = base_word_offset,
    };
    const uint64_t payload_bytes = static_cast<uint64_t>(node_count) *
                                   static_cast<uint64_t>(kActiveWordsPerNode + kOutputWordsPerNode) * sizeof(uint32_t);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    const bool source_valid = validate_source_values(source_words, max_nodes, node_count);

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        std::fill_n(output_words, node_count, kOutputSentinel);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            bool correctness_pass = dispatch_ok && source_valid;
            if (correctness_pass) {
                correctness_pass = std::equal(output_words, output_words + node_count, reference.begin());
            }
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << variant.variant_name << ", node_count=" << node_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        std::fill_n(output_words, node_count, kOutputSentinel);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && source_valid && std::equal(output_words, output_words + node_count, reference.begin());
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, variant, node_count, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << variant.variant_name << ", node_count=" << node_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant.variant_name,
            .problem_size = node_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(node_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + variant.variant_name + "_nodes_" + std::to_string(node_count),
        dispatch_samples));
    return true;
}

} // namespace

BvhNodeLayoutExperimentOutput run_bvh_node_layout_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                             const BvhNodeLayoutExperimentConfig& config) {
    BvhNodeLayoutExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "bvh node layout experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "36_bvh_node_layout.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for bvh node layout experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_nodes =
        determine_logical_count(config.max_buffer_bytes, properties.limits.maxComputeWorkGroupCount[0]);
    if (max_nodes == 0U) {
        std::cerr << "Scratch buffer too small for bvh node layout experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_nodes=" << max_nodes
                  << ", source_span_bytes=" << compute_source_span_bytes(max_nodes)
                  << ", output_span_bytes=" << compute_output_span_bytes(max_nodes)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_nodes, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* source_words = static_cast<uint32_t*>(buffers.source_mapped_ptr);
    if (source_words == nullptr || buffers.output_mapped_ptr == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for buffers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_source_values(source_words, max_nodes);

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_nodes);
    for (const auto& variant : kVariantDescriptors) {
        for (uint32_t node_count : problem_sizes) {
            if (!run_case(context, runner, pipeline, buffers, variant, node_count, max_nodes, output,
                          config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
