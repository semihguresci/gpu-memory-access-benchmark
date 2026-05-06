#include "experiments/frustum_vs_clustered_culling_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "38_frustum_vs_clustered_culling";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kClusterCount = 16U;
constexpr uint32_t kTargetEntityCount = 8388608U;
constexpr uint32_t kMinimumEntityCount = kWorkgroupSize;
constexpr uint32_t kVisibilityCounterCount = 1U + kClusterCount;
constexpr uint32_t kVisibleSentinel = 0xFFFFFFFFU;

enum class CullingMode : uint32_t {
    FrustumDirect = 0U,
    ClusteredCulling = 1U,
};

enum class DistributionKind : std::uint8_t {
    WideScene,
    CenterClustered,
};

struct alignas(16) EntityRecord {
    float position_x = 0.0F;
    float position_y = 0.0F;
    float position_z = 0.0F;
    float entity_id = 0.0F;
};

static_assert(sizeof(EntityRecord) == (sizeof(float) * 4U));

struct CullingDescriptor {
    CullingMode mode;
    const char* name;
};

struct DistributionDescriptor {
    DistributionKind kind;
    const char* name;
};

constexpr std::array<CullingDescriptor, 2> kCullingDescriptors = {{
    {CullingMode::FrustumDirect, "frustum_direct"},
    {CullingMode::ClusteredCulling, "clustered_culling"},
}};

constexpr std::array<DistributionDescriptor, 2> kDistributionDescriptors = {{
    {DistributionKind::WideScene, "wide_scene"},
    {DistributionKind::CenterClustered, "center_clustered"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource counters_buffer{};
    BufferResource visible_ids_buffer{};
    void* input_mapped_ptr = nullptr;
    void* counters_mapped_ptr = nullptr;
    void* visible_ids_mapped_ptr = nullptr;
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
    uint32_t entity_count = 0U;
    uint32_t culling_mode = 0U;
    uint32_t cluster_count = 0U;
    uint32_t max_visible_entries = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct PreparedCaseData {
    std::vector<EntityRecord> dispatch_records;
    std::vector<uint32_t> expected_visible_ids;
    std::vector<uint32_t> expected_cluster_counts;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

const char* culling_name(CullingMode mode) {
    for (const auto& descriptor : kCullingDescriptors) {
        if (descriptor.mode == mode) {
            return descriptor.name;
        }
    }
    return "unknown_culling";
}

const char* distribution_name(DistributionKind kind) {
    for (const auto& descriptor : kDistributionDescriptors) {
        if (descriptor.kind == kind) {
            return descriptor.name;
        }
    }
    return "unknown_distribution";
}

std::string make_variant_name(CullingMode mode, DistributionKind kind) {
    return std::string(culling_name(mode)) + "_" + distribution_name(kind);
}

std::string make_case_name(CullingMode mode, DistributionKind kind, uint32_t entity_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(mode, kind) + "_entities_" +
           std::to_string(entity_count);
}

VkDeviceSize compute_input_span_bytes(uint32_t entity_count) {
    return static_cast<VkDeviceSize>(entity_count) * static_cast<VkDeviceSize>(sizeof(EntityRecord));
}

VkDeviceSize compute_counter_span_bytes() {
    return static_cast<VkDeviceSize>(kVisibilityCounterCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_visible_span_bytes(uint32_t entity_count) {
    return static_cast<VkDeviceSize>(entity_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t determine_entity_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t input_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(EntityRecord);
    const uint64_t visible_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped =
        std::min({input_capacity, visible_capacity, dispatch_capacity, static_cast<uint64_t>(kTargetEntityCount)});
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kMinimumEntityCount) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_entities) {
    const std::array<uint32_t, 6> base_sizes = {262144U, 524288U, 1048576U, 2097152U, 4194304U, 8388608U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_entities) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_entities) {
        sizes.push_back(max_entities);
    }
    return sizes;
}

float sample_signed(std::mt19937& generator) {
    return (static_cast<float>(generator() & 0xFFFFU) / 32767.5F) - 1.0F;
}

float sample_unit(std::mt19937& generator) {
    return static_cast<float>(generator() & 0xFFFFU) / 65535.0F;
}

std::vector<EntityRecord> build_records(DistributionKind kind, uint32_t entity_count, uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<EntityRecord> records(entity_count);

    for (uint32_t index = 0U; index < entity_count; ++index) {
        float x = sample_signed(generator);
        float y = sample_signed(generator);
        float z = sample_unit(generator);

        if (kind == DistributionKind::CenterClustered) {
            x *= 0.45F;
            y *= 0.45F;
            z = std::clamp(0.35F + (z * 0.3F), 0.0F, 0.999999F);
        }

        records[index] = EntityRecord{
            .position_x = x,
            .position_y = y,
            .position_z = z,
            .entity_id = static_cast<float>(index),
        };
    }

    std::shuffle(records.begin(), records.end(), generator);
    return records;
}

bool is_visible(const EntityRecord& record) {
    return std::abs(record.position_x) <= 0.8F && std::abs(record.position_y) <= 0.8F && record.position_z <= 0.9F;
}

uint32_t compute_cluster_index(float position_z) {
    const float clamped = std::clamp(position_z, 0.0F, 0.999999F);
    const uint32_t scaled = static_cast<uint32_t>(clamped * static_cast<float>(kClusterCount));
    return std::min(scaled, kClusterCount - 1U);
}

PreparedCaseData prepare_case_data(CullingMode mode, DistributionKind kind, uint32_t entity_count, uint32_t seed) {
    PreparedCaseData prepared{};
    prepared.dispatch_records = build_records(kind, entity_count, seed);
    prepared.expected_cluster_counts.assign(kClusterCount, 0U);

    for (const EntityRecord& record : prepared.dispatch_records) {
        if (!is_visible(record)) {
            continue;
        }

        const uint32_t entity_id = static_cast<uint32_t>(record.entity_id);
        prepared.expected_visible_ids.push_back(entity_id);
        if (mode == CullingMode::ClusteredCulling) {
            ++prepared.expected_cluster_counts[compute_cluster_index(record.position_z)];
        }
    }

    std::sort(prepared.expected_visible_ids.begin(), prepared.expected_visible_ids.end());
    return prepared;
}

uint64_t compute_estimated_global_traffic_bytes(uint32_t entity_count, uint32_t visible_count, CullingMode mode) {
    const uint64_t input_bytes = static_cast<uint64_t>(entity_count) * sizeof(EntityRecord);
    const uint64_t visible_output_bytes = static_cast<uint64_t>(visible_count) * sizeof(uint32_t);
    uint64_t counter_atomic_bytes = static_cast<uint64_t>(visible_count) * sizeof(uint32_t);
    if (mode == CullingMode::ClusteredCulling) {
        counter_atomic_bytes += static_cast<uint64_t>(visible_count) * sizeof(uint32_t);
    }
    return input_bytes + visible_output_bytes + counter_atomic_bytes;
}

uint64_t compute_logical_payload_bytes(uint32_t entity_count, uint32_t visible_count) {
    const uint64_t input_bytes = static_cast<uint64_t>(entity_count) * sizeof(EntityRecord);
    const uint64_t visible_output_bytes = static_cast<uint64_t>(visible_count) * sizeof(uint32_t);
    return input_bytes + visible_output_bytes;
}

double compute_effective_gbps(uint32_t entity_count, uint32_t visible_count, CullingMode mode, double dispatch_ms) {
    return compute_effective_gbps_from_bytes(compute_estimated_global_traffic_bytes(entity_count, visible_count, mode),
                                             dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, uint32_t entity_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_input_span_bytes(entity_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_buffer)) {
        std::cerr << "Failed to create frustum-vs-clustered input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_counter_span_bytes(),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.counters_buffer)) {
        std::cerr << "Failed to create frustum-vs-clustered counters buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_visible_span_bytes(entity_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.visible_ids_buffer)) {
        std::cerr << "Failed to create frustum-vs-clustered visible-id buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.counters_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "frustum-vs-clustered input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.visible_ids_buffer);
        destroy_buffer_resource(context.device(), out_resources.counters_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.counters_buffer, "frustum-vs-clustered counters buffer",
                           out_resources.counters_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.visible_ids_buffer);
        destroy_buffer_resource(context.device(), out_resources.counters_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.visible_ids_buffer, "frustum-vs-clustered visible-id buffer",
                           out_resources.visible_ids_mapped_ptr)) {
        if (out_resources.counters_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.counters_buffer.memory);
            out_resources.counters_mapped_ptr = nullptr;
        }
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.visible_ids_buffer);
        destroy_buffer_resource(context.device(), out_resources.counters_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.visible_ids_mapped_ptr != nullptr && resources.visible_ids_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.visible_ids_buffer.memory);
        resources.visible_ids_mapped_ptr = nullptr;
    }
    if (resources.counters_mapped_ptr != nullptr && resources.counters_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.counters_buffer.memory);
        resources.counters_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.visible_ids_buffer);
    destroy_buffer_resource(context.device(), resources.counters_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo counter_info{buffers.counters_buffer.buffer, 0U, buffers.counters_buffer.size};
    const VkDescriptorBufferInfo visible_info{
        buffers.visible_ids_buffer.buffer,
        0U,
        buffers.visible_ids_buffer.size,
    };
    // Shader binding contract:
    //   0 -> entity records (xyz position, w entity id)
    //   1 -> counter bank: slot 0 visible count, slots [1..] cluster counts
    //   2 -> compact visible-id output stream
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = input_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = counter_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = visible_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load frustum-vs-clustered shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create frustum-vs-clustered descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create frustum-vs-clustered descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate frustum-vs-clustered descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create frustum-vs-clustered pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create frustum-vs-clustered pipeline.\n";
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

bool validate_input_values(const EntityRecord* input_values, const std::vector<EntityRecord>& expected) {
    if (input_values == nullptr || expected.empty()) {
        return false;
    }
    return std::equal(expected.begin(), expected.end(), input_values,
                      [](const EntityRecord& lhs, const EntityRecord& rhs) {
                          return lhs.position_x == rhs.position_x && lhs.position_y == rhs.position_y &&
                                 lhs.position_z == rhs.position_z && lhs.entity_id == rhs.entity_id;
                      });
}

bool validate_output_values(const uint32_t* counters, const uint32_t* visible_ids, uint32_t entity_count,
                            CullingMode mode, const PreparedCaseData& prepared) {
    if (counters == nullptr || visible_ids == nullptr) {
        return false;
    }

    const uint32_t observed_visible_count = counters[0];
    if (observed_visible_count != prepared.expected_visible_ids.size() || observed_visible_count > entity_count) {
        return false;
    }

    if (mode == CullingMode::ClusteredCulling) {
        for (uint32_t cluster = 0U; cluster < kClusterCount; ++cluster) {
            if (counters[1U + cluster] != prepared.expected_cluster_counts[cluster]) {
                return false;
            }
        }
    } else {
        for (uint32_t cluster = 0U; cluster < kClusterCount; ++cluster) {
            if (counters[1U + cluster] != 0U) {
                return false;
            }
        }
    }

    std::vector<uint32_t> observed_visible_ids(observed_visible_count);
    for (uint32_t index = 0U; index < observed_visible_count; ++index) {
        observed_visible_ids[index] = visible_ids[index];
    }
    std::sort(observed_visible_ids.begin(), observed_visible_ids.end());
    if (observed_visible_ids != prepared.expected_visible_ids) {
        return false;
    }

    for (uint32_t index = observed_visible_count; index < entity_count; ++index) {
        if (visible_ids[index] != kVisibleSentinel) {
            return false;
        }
    }

    return true;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources,
                    const PushConstants& constants) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(constants.entity_count, kWorkgroupSize);
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

void reset_case_buffers(uint32_t* counters, uint32_t* visible_ids, uint32_t entity_count) {
    std::fill_n(counters, kVisibilityCounterCount, 0U);
    std::fill_n(visible_ids, entity_count, kVisibleSentinel);
}

void record_case_notes(std::string& notes, CullingMode mode, DistributionKind distribution, uint32_t entity_count,
                       const PreparedCaseData& prepared, uint64_t logical_payload_bytes,
                       uint64_t estimated_global_total_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "culling_mode=" + std::string(culling_name(mode)));
    append_note(notes, "distribution=" + std::string(distribution_name(distribution)));
    append_note(notes, "entity_count=" + std::to_string(entity_count));
    append_note(notes, "cluster_count=" + std::to_string(kClusterCount));
    append_note(notes, "expected_visible_count=" + std::to_string(prepared.expected_visible_ids.size()));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "logical_payload_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "estimated_global_total_bytes=" + std::to_string(estimated_global_total_bytes));
    append_note(notes, "gbps_mode=estimated_global_bytes");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, CullingMode mode, DistributionKind distribution, uint32_t entity_count,
              uint32_t pattern_seed, FrustumVsClusteredCullingExperimentOutput& output, bool verbose_progress) {
    auto* input_values = static_cast<EntityRecord*>(buffers.input_mapped_ptr);
    auto* counters = static_cast<uint32_t*>(buffers.counters_mapped_ptr);
    auto* visible_ids = static_cast<uint32_t*>(buffers.visible_ids_mapped_ptr);
    if (input_values == nullptr || counters == nullptr || visible_ids == nullptr) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped buffers for variant=" << make_variant_name(mode, distribution) << ".\n";
        return false;
    }

    const PreparedCaseData prepared = prepare_case_data(mode, distribution, entity_count, pattern_seed);
    std::copy(prepared.dispatch_records.begin(), prepared.dispatch_records.end(), input_values);

    // culling_mode controls whether binding 1 is treated as a single visible
    // counter or as visible count plus per-cluster counters.
    const PushConstants constants{
        .entity_count = entity_count,
        .culling_mode = static_cast<uint32_t>(mode),
        .cluster_count = kClusterCount,
        .max_visible_entries = entity_count,
    };
    const uint32_t expected_visible_count = static_cast<uint32_t>(prepared.expected_visible_ids.size());
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(entity_count, expected_visible_count);
    const uint64_t estimated_global_total_bytes =
        compute_estimated_global_traffic_bytes(entity_count, expected_visible_count, mode);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));
    const bool input_valid = validate_input_values(input_values, prepared.dispatch_records);

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_case_buffers(counters, visible_ids, entity_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass = dispatch_ok && input_valid &&
                                          validate_output_values(counters, visible_ids, entity_count, mode, prepared);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", entity_count=" << entity_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        reset_case_buffers(counters, visible_ids, entity_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid && validate_output_values(counters, visible_ids, entity_count, mode, prepared);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, mode, distribution, entity_count, prepared, logical_payload_bytes,
                          estimated_global_total_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", entity_count=" << entity_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = make_variant_name(mode, distribution),
            .problem_size = entity_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(entity_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(entity_count, expected_visible_count, mode, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(make_case_name(mode, distribution, entity_count), dispatch_samples));
    return true;
}

} // namespace

FrustumVsClusteredCullingExperimentOutput
run_frustum_vs_clustered_culling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                            const FrustumVsClusteredCullingExperimentConfig& config) {
    FrustumVsClusteredCullingExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "frustum-vs-clustered culling experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "38_frustum_vs_clustered_culling.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for frustum-vs-clustered culling experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);

    const uint32_t entity_count =
        determine_entity_count(config.max_buffer_bytes, properties.limits.maxComputeWorkGroupCount[0]);
    if (entity_count == 0U) {
        std::cerr << "Scratch buffer too small for frustum-vs-clustered culling experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_entities=" << entity_count
                  << ", input_span_bytes=" << compute_input_span_bytes(entity_count)
                  << ", counter_span_bytes=" << compute_counter_span_bytes()
                  << ", visible_span_bytes=" << compute_visible_span_bytes(entity_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, entity_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(entity_count);
    for (const auto& culling : kCullingDescriptors) {
        for (const auto& distribution : kDistributionDescriptors) {
            for (uint32_t case_size : problem_sizes) {
                const uint32_t seed = config.pattern_seed ^ (static_cast<uint32_t>(culling.mode) * 0xA511E9B3U) ^
                                      (static_cast<uint32_t>(distribution.kind) * 0x9E3779B9U) ^ case_size;
                if (!run_case(context, runner, pipeline, buffers, culling.mode, distribution.kind, case_size, seed,
                              output, config.verbose_progress)) {
                    output.all_points_correct = false;
                }
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
