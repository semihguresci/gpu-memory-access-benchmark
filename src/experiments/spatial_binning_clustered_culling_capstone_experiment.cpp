#include "experiments/spatial_binning_clustered_culling_capstone_experiment.hpp"

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

constexpr const char* kExperimentId = "25_spatial_binning_clustered_culling_capstone";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kBinCount = 64U;
constexpr uint32_t kTargetEntityCount = 65536U;
constexpr uint32_t kMinimumEntityCount = kWorkgroupSize;
constexpr uint32_t kDenseActiveBins = 8U;
constexpr uint32_t kClusterCount = 4U;
constexpr uint32_t kSentinelEntityId = 0xFFFFFFFFU;

enum class StrategyKind : std::uint8_t {
    GlobalAppend,
    CoherentAppend,
};

enum class DistributionKind : std::uint8_t {
    UniformSparse,
    UniformDense,
    Clustered,
};

struct alignas(8) SpatialPointRecord {
    float position_x = 0.0F;
    uint32_t entity_id = 0U;
};

static_assert(sizeof(SpatialPointRecord) == (sizeof(float) + sizeof(uint32_t)));

struct StrategyDescriptor {
    StrategyKind kind;
    const char* name;
    bool host_sorted;
};

struct DistributionDescriptor {
    DistributionKind kind;
    const char* name;
};

constexpr std::array<StrategyDescriptor, 2> kStrategyDescriptors = {{
    {StrategyKind::GlobalAppend, "global_append", false},
    {StrategyKind::CoherentAppend, "coherent_append", true},
}};

constexpr std::array<DistributionDescriptor, 3> kDistributionDescriptors = {{
    {DistributionKind::UniformSparse, "uniform_sparse"},
    {DistributionKind::UniformDense, "uniform_dense"},
    {DistributionKind::Clustered, "clustered"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource bin_counts_buffer{};
    BufferResource bin_lists_buffer{};
    void* input_mapped_ptr = nullptr;
    void* bin_counts_mapped_ptr = nullptr;
    void* bin_lists_mapped_ptr = nullptr;
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
    uint32_t bin_count = 0U;
    uint32_t max_entries_per_bin = 0U;
    uint32_t reserved = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct DistributionStats {
    uint32_t active_bin_count = 0U;
    uint32_t max_bin_load = 0U;
    double mean_bin_load = 0.0;
};

struct PreparedCaseData {
    std::vector<SpatialPointRecord> dispatch_records;
    std::vector<std::vector<uint32_t>> expected_bin_members;
    DistributionStats stats{};
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

const char* strategy_name(StrategyKind strategy_kind) {
    for (const auto& descriptor : kStrategyDescriptors) {
        if (descriptor.kind == strategy_kind) {
            return descriptor.name;
        }
    }
    return "unknown_strategy";
}

bool strategy_host_sorted(StrategyKind strategy_kind) {
    for (const auto& descriptor : kStrategyDescriptors) {
        if (descriptor.kind == strategy_kind) {
            return descriptor.host_sorted;
        }
    }
    return false;
}

const char* distribution_name(DistributionKind distribution_kind) {
    for (const auto& descriptor : kDistributionDescriptors) {
        if (descriptor.kind == distribution_kind) {
            return descriptor.name;
        }
    }
    return "unknown_distribution";
}

std::string make_variant_name(StrategyKind strategy_kind, DistributionKind distribution_kind) {
    return std::string(strategy_name(strategy_kind)) + "_" + distribution_name(distribution_kind);
}

std::string make_case_name(StrategyKind strategy_kind, DistributionKind distribution_kind, uint32_t entity_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(strategy_kind, distribution_kind) + "_entities_" +
           std::to_string(entity_count);
}

uint32_t distribution_seed(const SpatialBinningClusteredCullingCapstoneExperimentConfig& config,
                           DistributionKind distribution_kind) {
    switch (distribution_kind) {
    case DistributionKind::UniformSparse:
        return config.pattern_seed ^ 0x13579BDFU;
    case DistributionKind::UniformDense:
        return config.pattern_seed ^ 0x2468ACE0U;
    case DistributionKind::Clustered:
        return config.pattern_seed ^ 0x55AA7733U;
    }
    return config.pattern_seed;
}

VkDeviceSize compute_input_span_bytes(uint32_t entity_count) {
    return static_cast<VkDeviceSize>(entity_count) * static_cast<VkDeviceSize>(sizeof(SpatialPointRecord));
}

VkDeviceSize compute_counter_span_bytes() {
    return static_cast<VkDeviceSize>(kBinCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_bin_list_span_bytes(uint32_t entity_count) {
    return static_cast<VkDeviceSize>(kBinCount) * static_cast<VkDeviceSize>(entity_count) *
           static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t determine_entity_count(std::size_t total_budget_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t fixed_counter_bytes = static_cast<uint64_t>(compute_counter_span_bytes());
    if (total_budget_bytes <= fixed_counter_bytes) {
        return 0U;
    }

    const uint64_t variable_budget_bytes = static_cast<uint64_t>(total_budget_bytes) - fixed_counter_bytes;
    const uint64_t bytes_per_entity =
        static_cast<uint64_t>(sizeof(SpatialPointRecord)) + (static_cast<uint64_t>(kBinCount) * sizeof(uint32_t));
    if (bytes_per_entity == 0U) {
        return 0U;
    }

    const uint64_t buffer_limited_count = variable_budget_bytes / bytes_per_entity;
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped_count =
        std::min({buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(kTargetEntityCount)});
    const uint64_t rounded_count = capped_count - (capped_count % kWorkgroupSize);
    if (rounded_count < kMinimumEntityCount) {
        return 0U;
    }

    return static_cast<uint32_t>(rounded_count);
}

float jitter_from_rng(std::mt19937& generator) {
    const uint32_t bits = generator();
    const float normalized = static_cast<float>(bits & 0xFFFFU) / 65535.0F;
    return 0.125F + (normalized * 0.75F);
}

float make_position_from_bin(uint32_t bin_index, std::mt19937& generator) {
    const float jitter = jitter_from_rng(generator);
    return (static_cast<float>(bin_index) + jitter) / static_cast<float>(kBinCount);
}

uint32_t compute_bin_index(float position_x) {
    const float one_minus_epsilon = std::nextafter(1.0F, 0.0F);
    const float clamped = std::clamp(position_x, 0.0F, one_minus_epsilon);
    const uint32_t scaled = static_cast<uint32_t>(clamped * static_cast<float>(kBinCount));
    return std::min(scaled, kBinCount - 1U);
}

std::vector<SpatialPointRecord> build_records(DistributionKind distribution_kind, uint32_t entity_count,
                                              uint32_t seed) {
    std::vector<SpatialPointRecord> records(entity_count);
    std::mt19937 generator(seed);

    constexpr std::array<uint32_t, kClusterCount> kClusterCenters = {8U, 21U, 39U, 54U};
    constexpr std::array<uint32_t, kClusterCount> kClusterThresholds = {450U, 750U, 920U, 1000U};

    for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
        uint32_t bin_index = 0U;
        switch (distribution_kind) {
        case DistributionKind::UniformSparse:
            bin_index = generator() % kBinCount;
            break;
        case DistributionKind::UniformDense:
            bin_index = ((generator() % kDenseActiveBins) + ((kBinCount - kDenseActiveBins) / 2U)) % kBinCount;
            break;
        case DistributionKind::Clustered: {
            const uint32_t selector = generator() % 1000U;
            uint32_t cluster_index = 0U;
            while (cluster_index + 1U < kClusterThresholds.size() && selector >= kClusterThresholds[cluster_index]) {
                ++cluster_index;
            }
            const uint32_t center = kClusterCenters[cluster_index];
            const int32_t offset = static_cast<int32_t>(generator() % 5U) - 2;
            const int32_t clamped =
                std::clamp(static_cast<int32_t>(center) + offset, 0, static_cast<int32_t>(kBinCount) - 1);
            bin_index = static_cast<uint32_t>(clamped);
            break;
        }
        }

        records[entity_index] = SpatialPointRecord{
            .position_x = make_position_from_bin(bin_index, generator),
            .entity_id = entity_index,
        };
    }

    std::shuffle(records.begin(), records.end(), generator);
    return records;
}

PreparedCaseData prepare_case_data(StrategyKind strategy_kind, DistributionKind distribution_kind,
                                   uint32_t entity_count, uint32_t seed) {
    PreparedCaseData prepared{};
    prepared.dispatch_records = build_records(distribution_kind, entity_count, seed);
    if (strategy_host_sorted(strategy_kind)) {
        std::stable_sort(prepared.dispatch_records.begin(), prepared.dispatch_records.end(),
                         [](const SpatialPointRecord& left, const SpatialPointRecord& right) {
                             const uint32_t left_bin = compute_bin_index(left.position_x);
                             const uint32_t right_bin = compute_bin_index(right.position_x);
                             if (left_bin != right_bin) {
                                 return left_bin < right_bin;
                             }
                             return left.entity_id < right.entity_id;
                         });
    }

    prepared.expected_bin_members.assign(kBinCount, {});

    uint32_t active_bins = 0U;
    uint32_t max_bin_load = 0U;
    uint64_t total_assignments = 0U;
    for (const SpatialPointRecord& record : prepared.dispatch_records) {
        prepared.expected_bin_members[compute_bin_index(record.position_x)].push_back(record.entity_id);
    }
    for (auto& bin_members : prepared.expected_bin_members) {
        std::sort(bin_members.begin(), bin_members.end());
        if (!bin_members.empty()) {
            ++active_bins;
        }
        max_bin_load = std::max(max_bin_load, static_cast<uint32_t>(bin_members.size()));
        total_assignments += bin_members.size();
    }

    prepared.stats.active_bin_count = active_bins;
    prepared.stats.max_bin_load = max_bin_load;
    prepared.stats.mean_bin_load =
        active_bins == 0U ? 0.0 : static_cast<double>(total_assignments) / static_cast<double>(active_bins);
    return prepared;
}

uint64_t compute_estimated_global_traffic_bytes(uint32_t entity_count) {
    const uint64_t input_bytes = static_cast<uint64_t>(entity_count) * sizeof(SpatialPointRecord);
    const uint64_t counter_atomic_bytes = static_cast<uint64_t>(entity_count) * sizeof(uint32_t);
    const uint64_t output_bytes = static_cast<uint64_t>(entity_count) * sizeof(uint32_t);
    return input_bytes + counter_atomic_bytes + output_bytes;
}

double compute_effective_gbps(uint32_t entity_count, double dispatch_ms) {
    return compute_effective_gbps_from_bytes(compute_estimated_global_traffic_bytes(entity_count), dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, uint32_t entity_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_input_span_bytes(entity_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.input_buffer)) {
        std::cerr << "Failed to create spatial binning input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_counter_span_bytes(),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.bin_counts_buffer)) {
        std::cerr << "Failed to create spatial binning count buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_bin_list_span_bytes(entity_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.bin_lists_buffer)) {
        std::cerr << "Failed to create spatial binning list buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.bin_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "spatial binning input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.bin_lists_buffer);
        destroy_buffer_resource(context.device(), out_resources.bin_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.bin_counts_buffer, "spatial binning count buffer",
                           out_resources.bin_counts_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.bin_lists_buffer);
        destroy_buffer_resource(context.device(), out_resources.bin_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.bin_lists_buffer, "spatial binning list buffer",
                           out_resources.bin_lists_mapped_ptr)) {
        if (out_resources.bin_counts_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.bin_counts_buffer.memory);
            out_resources.bin_counts_mapped_ptr = nullptr;
        }
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.bin_lists_buffer);
        destroy_buffer_resource(context.device(), out_resources.bin_counts_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.bin_lists_mapped_ptr != nullptr && resources.bin_lists_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.bin_lists_buffer.memory);
        resources.bin_lists_mapped_ptr = nullptr;
    }
    if (resources.bin_counts_mapped_ptr != nullptr && resources.bin_counts_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.bin_counts_buffer.memory);
        resources.bin_counts_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.bin_lists_buffer);
    destroy_buffer_resource(context.device(), resources.bin_counts_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo count_info{buffers.bin_counts_buffer.buffer, 0U, buffers.bin_counts_buffer.size};
    const VkDescriptorBufferInfo list_info{buffers.bin_lists_buffer.buffer, 0U, buffers.bin_lists_buffer.size};

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
                                                              .buffer_info = count_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = list_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load spatial binning shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create spatial binning descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create spatial binning descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate spatial binning descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create spatial binning pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create spatial binning compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t entity_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(entity_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants push_constants{
        .entity_count = entity_count,
        .bin_count = kBinCount,
        .max_entries_per_bin = entity_count,
        .reserved = 0U,
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

void reset_output_buffers(uint32_t* bin_counts, uint32_t* bin_lists, uint32_t entity_count) {
    std::fill_n(bin_counts, kBinCount, 0U);
    std::fill_n(bin_lists, static_cast<std::size_t>(kBinCount) * entity_count, kSentinelEntityId);
}

bool validate_results(const uint32_t* actual_bin_counts, const uint32_t* actual_bin_lists, uint32_t entity_count,
                      const PreparedCaseData& prepared_case) {
    for (uint32_t bin_index = 0U; bin_index < kBinCount; ++bin_index) {
        const auto& expected_members = prepared_case.expected_bin_members[bin_index];
        const uint32_t actual_count = actual_bin_counts[bin_index];
        if (actual_count != expected_members.size() || actual_count > entity_count) {
            return false;
        }

        std::vector<uint32_t> actual_members(actual_count, 0U);
        const std::size_t base_index = static_cast<std::size_t>(bin_index) * entity_count;
        for (uint32_t entry_index = 0U; entry_index < actual_count; ++entry_index) {
            actual_members[entry_index] = actual_bin_lists[base_index + entry_index];
        }
        std::sort(actual_members.begin(), actual_members.end());
        if (actual_members != expected_members) {
            return false;
        }

        const std::size_t sentinel_checks = std::min<std::size_t>(3U, entity_count - actual_count);
        for (std::size_t tail_index = 0; tail_index < sentinel_checks; ++tail_index) {
            const std::size_t actual_index = base_index + actual_count + tail_index;
            if (actual_bin_lists[actual_index] != kSentinelEntityId) {
                return false;
            }
        }
    }

    return true;
}

void record_case_notes(std::string& notes, StrategyKind strategy_kind, DistributionKind distribution_kind,
                       const SpatialBinningClusteredCullingCapstoneExperimentConfig& config, uint32_t entity_count,
                       const DistributionStats& stats, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, std::string("strategy=") + strategy_name(strategy_kind));
    append_note(notes, std::string("distribution=") + distribution_name(distribution_kind));
    append_note(notes, "seed=" + std::to_string(distribution_seed(config, distribution_kind)));
    append_note(notes, "bin_count=" + std::to_string(kBinCount));
    append_note(notes, "active_bin_count=" + std::to_string(stats.active_bin_count));
    append_note(notes, "max_bin_load=" + std::to_string(stats.max_bin_load));
    append_note(notes, "mean_active_bin_load=" + std::to_string(stats.mean_bin_load));
    append_note(notes, "host_sorted=" + std::to_string(strategy_host_sorted(strategy_kind) ? 1U : 0U));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" +
                           std::to_string(VulkanComputeUtils::compute_group_count_1d(entity_count, kWorkgroupSize)));
    append_note(notes, "entity_count=" + std::to_string(entity_count));
    append_note(notes, "input_span_bytes=" + std::to_string(compute_input_span_bytes(entity_count)));
    append_note(notes, "counter_span_bytes=" + std::to_string(compute_counter_span_bytes()));
    append_note(notes, "bin_list_span_bytes=" + std::to_string(compute_bin_list_span_bytes(entity_count)));
    append_note(notes,
                "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(config.max_buffer_bytes)));
    append_note(notes,
                "estimated_global_total_bytes=" + std::to_string(compute_estimated_global_traffic_bytes(entity_count)));
    append_note(notes, "validation_mode=per_bin_sorted_entity_ids");
    append_note(notes, "append_order=nondeterministic_atomic");
    append_note(notes, "overflow_protection=per_bin_capacity_equals_entity_count");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, StrategyKind strategy_kind,
              DistributionKind distribution_kind, const SpatialBinningClusteredCullingCapstoneExperimentConfig& config,
              uint32_t entity_count, SpatialBinningClusteredCullingCapstoneExperimentOutput& output) {
    const PreparedCaseData prepared_case =
        prepare_case_data(strategy_kind, distribution_kind, entity_count, distribution_seed(config, distribution_kind));

    auto* input_records = static_cast<SpatialPointRecord*>(buffers.input_mapped_ptr);
    auto* bin_counts = static_cast<uint32_t*>(buffers.bin_counts_mapped_ptr);
    auto* bin_lists = static_cast<uint32_t*>(buffers.bin_lists_mapped_ptr);
    if (input_records == nullptr || bin_counts == nullptr || bin_lists == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for case "
                  << make_variant_name(strategy_kind, distribution_kind) << ".\n";
        return false;
    }

    std::memcpy(input_records, prepared_case.dispatch_records.data(),
                static_cast<std::size_t>(entity_count) * sizeof(SpatialPointRecord));

    const std::string variant_name = make_variant_name(strategy_kind, distribution_kind);
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", entity_count=" << entity_count << ", active_bin_count=" << prepared_case.stats.active_bin_count
                  << ", max_bin_load=" << prepared_case.stats.max_bin_load
                  << ", host_sorted=" << (strategy_host_sorted(strategy_kind) ? "true" : "false")
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_output_buffers(bin_counts, bin_lists, entity_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, entity_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_results(bin_counts, bin_lists, entity_count, prepared_case);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        reset_output_buffers(bin_counts, bin_lists, entity_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, entity_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_results(bin_counts, bin_lists, entity_count, prepared_case);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;
        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;

        std::string notes;
        record_case_notes(notes, strategy_kind, distribution_kind, config, entity_count, prepared_case.stats,
                          correctness_pass, dispatch_ok);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = entity_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(entity_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(entity_count, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
        dispatch_samples.push_back(dispatch_ms);
    }

    const BenchmarkResult summary = BenchmarkRunner::summarize_samples(
        make_case_name(strategy_kind, distribution_kind, entity_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", samples=" << summary.sample_count << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

SpatialBinningClusteredCullingCapstoneExperimentOutput run_spatial_binning_clustered_culling_capstone_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const SpatialBinningClusteredCullingCapstoneExperimentConfig& config) {
    SpatialBinningClusteredCullingCapstoneExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "spatial binning clustered culling capstone experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string append_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.append_shader_path, "25_spatial_binning_clustered_culling_capstone_append.comp.spv");
    if (append_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for spatial binning clustered culling capstone experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t entity_count =
        determine_entity_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (entity_count == 0U) {
        std::cerr << "Scratch buffer too small for spatial binning clustered culling capstone experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] append shader: " << append_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] entity_count=" << entity_count
                  << ", input_span_bytes=" << compute_input_span_bytes(entity_count)
                  << ", counter_span_bytes=" << compute_counter_span_bytes()
                  << ", bin_list_span_bytes=" << compute_bin_list_span_bytes(entity_count)
                  << ", scratch_size_bytes=" << config.max_buffer_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, entity_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, append_shader_path, buffers, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    for (const auto& distribution_descriptor : kDistributionDescriptors) {
        for (const auto& strategy_descriptor : kStrategyDescriptors) {
            if (!run_case(context, runner, buffers, pipeline_resources, strategy_descriptor.kind,
                          distribution_descriptor.kind, config, entity_count, output)) {
                output.all_points_correct = false;
            }
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
