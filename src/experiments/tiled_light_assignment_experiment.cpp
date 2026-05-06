#include "experiments/tiled_light_assignment_experiment.hpp"

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

constexpr const char* kExperimentId = "39_tiled_light_assignment";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTileCountX = 16U;
constexpr uint32_t kTileCountY = 16U;
constexpr uint32_t kTileCount = kTileCountX * kTileCountY;
constexpr uint32_t kTargetLightCount = 8388608U;
constexpr uint32_t kMinimumLightCount = 4096U;
constexpr uint32_t kStatsWordCount = 2U;
constexpr uint32_t kStatsModeSentinel = 0xFFFFFFFFU;
constexpr float kIntersectionDistanceEpsilon = 1.0e-6F;
constexpr float kStableRadiusBias = 1.0e-5F;

enum class AssignmentMode : uint32_t {
    // One shader invocation per light. Fast when overlap lists are sparse, but
    // overlapping lights contend on the same tile counters.
    LightAtomic = 0U,
    // One shader invocation per tile. Avoids tile-counter contention by scanning
    // all lights serially for each tile.
    TileSerial = 1U,
    // One workgroup per tile. Lanes split the light scan and reduce through
    // shared memory before writing one count.
    TileParallelShared = 2U,
};

enum class DistributionKind : std::uint8_t {
    UniformLights,
    CenterClustered,
};

struct alignas(16) LightRecord {
    float position_x = 0.0F;
    float position_y = 0.0F;
    float radius = 0.0F;
    float intensity = 0.0F;
};

static_assert(sizeof(LightRecord) == (sizeof(float) * 4U));

struct AssignmentDescriptor {
    AssignmentMode mode;
    const char* name;
};

struct DistributionDescriptor {
    DistributionKind kind;
    const char* name;
};

constexpr std::array<AssignmentDescriptor, 3> kAssignmentDescriptors = {{
    {AssignmentMode::LightAtomic, "light_atomic"},
    {AssignmentMode::TileSerial, "tile_serial"},
    {AssignmentMode::TileParallelShared, "tile_parallel_shared"},
}};

constexpr std::array<DistributionDescriptor, 2> kDistributionDescriptors = {{
    {DistributionKind::UniformLights, "uniform_lights"},
    {DistributionKind::CenterClustered, "center_clustered"},
}};

struct BufferResources {
    BufferResource light_buffer{};
    BufferResource tile_count_buffer{};
    BufferResource stats_buffer{};
    void* light_mapped_ptr = nullptr;
    void* tile_count_mapped_ptr = nullptr;
    void* stats_mapped_ptr = nullptr;
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
    uint32_t light_count = 0U;
    uint32_t tile_count_x = 0U;
    uint32_t tile_count_y = 0U;
    uint32_t assignment_mode = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct PreparedCaseData {
    // CPU oracle for the tile_count_buffer produced by all three shader modes.
    std::vector<uint32_t> expected_tile_counts;
    // CPU oracle for stats_buffer[0], the total number of light/tile overlaps.
    uint64_t expected_total_overlaps = 0U;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

const char* assignment_name(AssignmentMode mode) {
    for (const auto& descriptor : kAssignmentDescriptors) {
        if (descriptor.mode == mode) {
            return descriptor.name;
        }
    }
    return "unknown_assignment";
}

const char* distribution_name(DistributionKind kind) {
    for (const auto& descriptor : kDistributionDescriptors) {
        if (descriptor.kind == kind) {
            return descriptor.name;
        }
    }
    return "unknown_distribution";
}

std::string make_variant_name(AssignmentMode mode, DistributionKind kind) {
    return std::string(assignment_name(mode)) + "_" + distribution_name(kind);
}

std::string make_case_name(AssignmentMode mode, DistributionKind kind, uint32_t light_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(mode, kind) + "_lights_" + std::to_string(light_count);
}

VkDeviceSize compute_light_span_bytes(uint32_t light_count) {
    return static_cast<VkDeviceSize>(light_count) * static_cast<VkDeviceSize>(sizeof(LightRecord));
}

VkDeviceSize compute_tile_count_span_bytes() {
    return static_cast<VkDeviceSize>(kTileCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_stats_span_bytes() {
    return static_cast<VkDeviceSize>(kStatsWordCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t determine_light_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t light_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(LightRecord);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped = std::min({light_capacity, dispatch_capacity, static_cast<uint64_t>(kTargetLightCount)});
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kMinimumLightCount) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_lights) {
    const std::array<uint32_t, 6> base_sizes = {262144U, 524288U, 1048576U, 2097152U, 4194304U, 8388608U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_lights) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_lights) {
        sizes.push_back(max_lights);
    }
    return sizes;
}

float sample_signed(std::mt19937& generator) {
    return (static_cast<float>(generator() & 0xFFFFU) / 32767.5F) - 1.0F;
}

float sample_unit(std::mt19937& generator) {
    return static_cast<float>(generator() & 0xFFFFU) / 65535.0F;
}

std::vector<LightRecord> build_lights(DistributionKind distribution, uint32_t light_count, uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<LightRecord> lights(light_count);

    for (uint32_t index = 0U; index < light_count; ++index) {
        // Uniform lights approximate a broad scene. Center-clustered lights
        // deliberately stress atomic contention in the light-centric path.
        float x = sample_unit(generator);
        float y = sample_unit(generator);
        float radius = 0.02F + (sample_unit(generator) * 0.09F);

        if (distribution == DistributionKind::CenterClustered) {
            if (sample_unit(generator) < 0.75F) {
                x = std::clamp(0.5F + (sample_signed(generator) * 0.18F), 0.0F, 1.0F);
                y = std::clamp(0.5F + (sample_signed(generator) * 0.18F), 0.0F, 1.0F);
            }
            radius = 0.03F + (sample_unit(generator) * 0.10F);
        }

        // Apply a tiny inward bias so exact tangent cases do not flip between
        // the CPU oracle and GPU shader at large light counts.
        const float stable_radius = std::max(radius - kStableRadiusBias, 0.0F);
        lights[index] = LightRecord{
            .position_x = x,
            .position_y = y,
            .radius = stable_radius,
            .intensity = 0.5F + (sample_unit(generator) * 2.0F),
        };
    }

    std::shuffle(lights.begin(), lights.end(), generator);
    return lights;
}

bool circle_intersects_tile(const LightRecord& light, uint32_t tile_x, uint32_t tile_y) {
    const float tile_width = 1.0F / static_cast<float>(kTileCountX);
    const float tile_height = 1.0F / static_cast<float>(kTileCountY);

    const float min_x = static_cast<float>(tile_x) * tile_width;
    const float min_y = static_cast<float>(tile_y) * tile_height;
    const float max_x = min_x + tile_width;
    const float max_y = min_y + tile_height;

    const float closest_x = std::clamp(light.position_x, min_x, max_x);
    const float closest_y = std::clamp(light.position_y, min_y, max_y);
    const float dx = light.position_x - closest_x;
    const float dy = light.position_y - closest_y;
    return (dx * dx + dy * dy) <= ((light.radius * light.radius) + kIntersectionDistanceEpsilon);
}

PreparedCaseData prepare_case_data(const std::vector<LightRecord>& lights, uint32_t light_count) {
    PreparedCaseData prepared{};
    prepared.expected_tile_counts.assign(kTileCount, 0U);
    prepared.expected_total_overlaps = 0U;

    // CPU oracle mirrors the shader's exact circle/tile test. It intentionally
    // produces only final counts, not lists, because the experiment focuses on
    // assignment strategy cost and contention rather than list storage layout.
    for (uint32_t index = 0U; index < light_count; ++index) {
        const LightRecord& light = lights[index];
        const float min_x = light.position_x - light.radius;
        const float max_x = light.position_x + light.radius;
        const float min_y = light.position_y - light.radius;
        const float max_y = light.position_y + light.radius;

        int min_tile_x = static_cast<int>(std::floor(min_x * static_cast<float>(kTileCountX)));
        int max_tile_x = static_cast<int>(std::floor(max_x * static_cast<float>(kTileCountX)));
        int min_tile_y = static_cast<int>(std::floor(min_y * static_cast<float>(kTileCountY)));
        int max_tile_y = static_cast<int>(std::floor(max_y * static_cast<float>(kTileCountY)));

        if (max_tile_x < 0 || max_tile_y < 0) {
            continue;
        }
        if (min_tile_x >= static_cast<int>(kTileCountX) || min_tile_y >= static_cast<int>(kTileCountY)) {
            continue;
        }

        // The light-driven path uses a bounding-box prepass before the exact
        // circle/tile test. Expand by one tile so host and shader stay aligned
        // even when a light edge lands close to a tile boundary.
        min_tile_x = std::clamp(min_tile_x - 1, 0, static_cast<int>(kTileCountX) - 1);
        max_tile_x = std::clamp(max_tile_x + 1, 0, static_cast<int>(kTileCountX) - 1);
        min_tile_y = std::clamp(min_tile_y - 1, 0, static_cast<int>(kTileCountY) - 1);
        max_tile_y = std::clamp(max_tile_y + 1, 0, static_cast<int>(kTileCountY) - 1);

        for (int tile_y = min_tile_y; tile_y <= max_tile_y; ++tile_y) {
            for (int tile_x = min_tile_x; tile_x <= max_tile_x; ++tile_x) {
                const uint32_t tile_x_u32 = static_cast<uint32_t>(tile_x);
                const uint32_t tile_y_u32 = static_cast<uint32_t>(tile_y);
                if (!circle_intersects_tile(light, tile_x_u32, tile_y_u32)) {
                    continue;
                }
                const uint32_t tile_index = tile_y_u32 * kTileCountX + tile_x_u32;
                ++prepared.expected_tile_counts[tile_index];
                ++prepared.expected_total_overlaps;
            }
        }
    }

    return prepared;
}

uint64_t compute_estimated_global_traffic_bytes(uint32_t light_count, uint64_t overlap_count, AssignmentMode mode) {
    if (mode == AssignmentMode::LightAtomic) {
        // Light-centric path reads each light once and writes one tile counter
        // plus one stats counter for every actual overlap.
        const uint64_t read_light_bytes = static_cast<uint64_t>(light_count) * sizeof(LightRecord);
        const uint64_t overlap_atomic_bytes = overlap_count * sizeof(uint32_t) * 2ULL;
        return read_light_bytes + overlap_atomic_bytes;
    }

    // Tile-centric paths scan the whole light list once per tile. This estimate
    // is deliberately larger than the logical payload and is the metric exported
    // as GB/s for apples-to-apples runtime interpretation.
    const uint64_t read_light_bytes =
        static_cast<uint64_t>(light_count) * static_cast<uint64_t>(kTileCount) * sizeof(LightRecord);
    const uint64_t tile_output_bytes = static_cast<uint64_t>(kTileCount) * sizeof(uint32_t);
    const uint64_t stats_atomic_bytes = static_cast<uint64_t>(kTileCount) * sizeof(uint32_t);
    return read_light_bytes + tile_output_bytes + stats_atomic_bytes;
}

uint64_t compute_logical_payload_bytes(uint32_t light_count) {
    const uint64_t light_input_bytes = static_cast<uint64_t>(light_count) * sizeof(LightRecord);
    const uint64_t tile_output_bytes = static_cast<uint64_t>(kTileCount) * sizeof(uint32_t);
    const uint64_t stats_output_bytes = sizeof(uint32_t);
    return light_input_bytes + tile_output_bytes + stats_output_bytes;
}

double compute_effective_gbps(uint32_t light_count, uint64_t overlap_count, AssignmentMode mode, double dispatch_ms) {
    return compute_effective_gbps_from_bytes(compute_estimated_global_traffic_bytes(light_count, overlap_count, mode),
                                             dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_light_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_light_span_bytes(max_light_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.light_buffer)) {
        std::cerr << "Failed to create tiled-light-assignment light buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_tile_count_span_bytes(),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.tile_count_buffer)) {
        std::cerr << "Failed to create tiled-light-assignment tile-count buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.light_buffer);
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), compute_stats_span_bytes(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.stats_buffer)) {
        std::cerr << "Failed to create tiled-light-assignment stats buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.tile_count_buffer);
        destroy_buffer_resource(context.device(), out_resources.light_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.light_buffer, "tiled-light-assignment light buffer",
                           out_resources.light_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.stats_buffer);
        destroy_buffer_resource(context.device(), out_resources.tile_count_buffer);
        destroy_buffer_resource(context.device(), out_resources.light_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.tile_count_buffer, "tiled-light-assignment tile-count buffer",
                           out_resources.tile_count_mapped_ptr)) {
        if (out_resources.light_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.light_buffer.memory);
            out_resources.light_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.stats_buffer);
        destroy_buffer_resource(context.device(), out_resources.tile_count_buffer);
        destroy_buffer_resource(context.device(), out_resources.light_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.stats_buffer, "tiled-light-assignment stats buffer",
                           out_resources.stats_mapped_ptr)) {
        if (out_resources.tile_count_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.tile_count_buffer.memory);
            out_resources.tile_count_mapped_ptr = nullptr;
        }
        if (out_resources.light_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.light_buffer.memory);
            out_resources.light_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.stats_buffer);
        destroy_buffer_resource(context.device(), out_resources.tile_count_buffer);
        destroy_buffer_resource(context.device(), out_resources.light_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.stats_mapped_ptr != nullptr && resources.stats_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.stats_buffer.memory);
        resources.stats_mapped_ptr = nullptr;
    }
    if (resources.tile_count_mapped_ptr != nullptr && resources.tile_count_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.tile_count_buffer.memory);
        resources.tile_count_mapped_ptr = nullptr;
    }
    if (resources.light_mapped_ptr != nullptr && resources.light_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.light_buffer.memory);
        resources.light_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.stats_buffer);
    destroy_buffer_resource(context.device(), resources.tile_count_buffer);
    destroy_buffer_resource(context.device(), resources.light_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo light_info{buffers.light_buffer.buffer, 0U, buffers.light_buffer.size};
    const VkDescriptorBufferInfo tile_count_info{
        buffers.tile_count_buffer.buffer,
        0U,
        buffers.tile_count_buffer.size,
    };
    const VkDescriptorBufferInfo stats_info{buffers.stats_buffer.buffer, 0U, buffers.stats_buffer.size};

    // Shader binding contract:
    //   0 -> light records consumed by every assignment path
    //   1 -> one tile counter per screen tile
    //   2 -> stats[0] overlap total, stats[1] assignment-mode echo
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = light_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = tile_count_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = stats_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load tiled-light-assignment shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create tiled-light-assignment descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create tiled-light-assignment descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate tiled-light-assignment descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create tiled-light-assignment pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create tiled-light-assignment compute pipeline.\n";
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

bool validate_input_values(const LightRecord* input_values, const std::vector<LightRecord>& expected,
                           uint32_t light_count) {
    if (input_values == nullptr || expected.size() < light_count) {
        return false;
    }
    return std::memcmp(input_values, expected.data(), static_cast<size_t>(light_count) * sizeof(LightRecord)) == 0;
}

bool validate_output_values(const uint32_t* tile_counts, const uint32_t* stats, const PreparedCaseData& prepared,
                            AssignmentMode mode) {
    if (tile_counts == nullptr || stats == nullptr) {
        return false;
    }

    for (uint32_t tile = 0U; tile < kTileCount; ++tile) {
        if (tile_counts[tile] != prepared.expected_tile_counts[tile]) {
            return false;
        }
    }
    if (static_cast<uint64_t>(stats[0]) != prepared.expected_total_overlaps) {
        return false;
    }
    if (stats[1] != static_cast<uint32_t>(mode)) {
        return false;
    }
    return true;
}

std::string describe_validation_failure(const uint32_t* tile_counts, const uint32_t* stats,
                                        const PreparedCaseData& prepared, AssignmentMode mode) {
    if (tile_counts == nullptr || stats == nullptr) {
        return "null_output_buffer";
    }

    for (uint32_t tile = 0U; tile < kTileCount; ++tile) {
        if (tile_counts[tile] != prepared.expected_tile_counts[tile]) {
            return "tile_" + std::to_string(tile) + "_expected_" + std::to_string(prepared.expected_tile_counts[tile]) +
                   "_actual_" + std::to_string(tile_counts[tile]);
        }
    }

    if (static_cast<uint64_t>(stats[0]) != prepared.expected_total_overlaps) {
        return "overlap_total_expected_" + std::to_string(prepared.expected_total_overlaps) + "_actual_" +
               std::to_string(stats[0]);
    }

    if (stats[1] != static_cast<uint32_t>(mode)) {
        return "mode_echo_expected_" + std::to_string(static_cast<uint32_t>(mode)) + "_actual_" +
               std::to_string(stats[1]);
    }

    return "unknown_validation_mismatch";
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, const PushConstants& constants,
                    uint32_t max_dispatch_groups_x) {
    const uint32_t tile_count = constants.tile_count_x * constants.tile_count_y;
    uint32_t group_count_x = 0U;
    // assignment_mode also determines dispatch geometry because each shader path
    // maps global ids to work differently:
    //   LightAtomic: one invocation per light
    //   TileSerial: one invocation per tile
    //   TileParallelShared: one workgroup per tile
    switch (static_cast<AssignmentMode>(constants.assignment_mode)) {
    case AssignmentMode::LightAtomic:
        group_count_x = VulkanComputeUtils::compute_group_count_1d(constants.light_count, kWorkgroupSize);
        break;
    case AssignmentMode::TileSerial:
        group_count_x = VulkanComputeUtils::compute_group_count_1d(tile_count, kWorkgroupSize);
        break;
    case AssignmentMode::TileParallelShared:
        group_count_x = tile_count;
        break;
    }

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

void reset_case_buffers(uint32_t* tile_counts, uint32_t* stats) {
    std::fill_n(tile_counts, kTileCount, 0U);
    stats[0] = 0U;
    stats[1] = kStatsModeSentinel;
}

void record_case_notes(std::string& notes, AssignmentMode mode, DistributionKind distribution, uint32_t light_count,
                       const PreparedCaseData& prepared, uint64_t logical_payload_bytes,
                       uint64_t estimated_global_total_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "assignment_mode=" + std::string(assignment_name(mode)));
    append_note(notes, "distribution=" + std::string(distribution_name(distribution)));
    append_note(notes, "light_count=" + std::to_string(light_count));
    append_note(notes, "tile_count_x=" + std::to_string(kTileCountX));
    append_note(notes, "tile_count_y=" + std::to_string(kTileCountY));
    append_note(notes, "expected_total_overlaps=" + std::to_string(prepared.expected_total_overlaps));
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
              const BufferResources& buffers, AssignmentMode mode, DistributionKind distribution, uint32_t light_count,
              const std::vector<LightRecord>& case_lights, const PreparedCaseData& prepared,
              uint32_t max_dispatch_groups_x, TiledLightAssignmentExperimentOutput& output, bool verbose_progress) {
    auto* light_values = static_cast<LightRecord*>(buffers.light_mapped_ptr);
    auto* tile_counts = static_cast<uint32_t*>(buffers.tile_count_mapped_ptr);
    auto* stats = static_cast<uint32_t*>(buffers.stats_mapped_ptr);
    if (light_values == nullptr || tile_counts == nullptr || stats == nullptr) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped buffers for variant=" << make_variant_name(mode, distribution) << ".\n";
        return false;
    }

    std::copy_n(case_lights.begin(), light_count, light_values);
    const bool input_valid = validate_input_values(light_values, case_lights, light_count);

    // assignment_mode values match the shader branches:
    //   0 = per-light atomic update
    //   1 = per-tile serial walk over all lights
    //   2 = per-tile shared-memory reduction
    const PushConstants constants{
        .light_count = light_count,
        .tile_count_x = kTileCountX,
        .tile_count_y = kTileCountY,
        .assignment_mode = static_cast<uint32_t>(mode),
    };
    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(light_count);
    const uint64_t estimated_global_total_bytes =
        compute_estimated_global_traffic_bytes(light_count, prepared.expected_total_overlaps, mode);

    // Each timed sample starts from zeroed counters so atomic/list-building work
    // is measured independently and validation can detect stale output.
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_case_buffers(tile_counts, stats);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass =
                dispatch_ok && input_valid && validate_output_values(tile_counts, stats, prepared, mode);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", light_count=" << light_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        reset_case_buffers(tile_counts, stats);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, constants, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid && validate_output_values(tile_counts, stats, prepared, mode);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, mode, distribution, light_count, prepared, logical_payload_bytes,
                          estimated_global_total_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", light_count=" << light_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail");
            if (!correctness_pass) {
                std::cout << ", failure=" << describe_validation_failure(tile_counts, stats, prepared, mode);
            }
            std::cout << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = make_variant_name(mode, distribution),
            .problem_size = light_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(light_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(light_count, prepared.expected_total_overlaps, mode, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(make_case_name(mode, distribution, light_count), dispatch_samples));
    return true;
}

} // namespace

TiledLightAssignmentExperimentOutput
run_tiled_light_assignment_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                      const TiledLightAssignmentExperimentConfig& config) {
    TiledLightAssignmentExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "tiled-light-assignment experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "39_tiled_light_assignment.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for tiled-light-assignment experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    if (kTileCount > max_dispatch_groups_x) {
        std::cerr << "Device maxComputeWorkGroupCount[0] is too small for tiled-light-assignment mode "
                     "tile_parallel_shared.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t max_light_count = determine_light_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_light_count == 0U) {
        std::cerr << "Scratch buffer too small for tiled-light-assignment experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_lights=" << max_light_count
                  << ", light_span_bytes=" << compute_light_span_bytes(max_light_count)
                  << ", tile_count_span_bytes=" << compute_tile_count_span_bytes()
                  << ", stats_span_bytes=" << compute_stats_span_bytes() << ", tile_count=" << kTileCount
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_light_count, buffers)) {
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

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_light_count);
    for (const auto& distribution : kDistributionDescriptors) {
        const uint32_t distribution_seed =
            config.pattern_seed ^ (static_cast<uint32_t>(distribution.kind) * 0x9E3779B9U);
        // Build the largest deterministic light set once per distribution. Each
        // smaller problem size uses a prefix of the same distribution so scaling
        // trends are not polluted by a different random scene.
        const std::vector<LightRecord> case_lights =
            build_lights(distribution.kind, max_light_count, distribution_seed);
        for (uint32_t light_count : problem_sizes) {
            const PreparedCaseData prepared = prepare_case_data(case_lights, light_count);
            for (const auto& assignment : kAssignmentDescriptors) {
                if (!run_case(context, runner, pipeline, buffers, assignment.mode, distribution.kind, light_count,
                              case_lights, prepared, max_dispatch_groups_x, output, config.verbose_progress)) {
                    output.all_points_correct = false;
                }
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
