
#include "experiments/std430_std140_packed_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/layout_assert.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ranges>
#include <string>
#include <vector>

namespace {

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "08_std430_std140_packed";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr float kValidationEpsilon = 1.0e-5F;
constexpr uint32_t kPackedFloatsPerParticle = 16U;
constexpr VkDeviceSize kPackedStorageStrideBytes = static_cast<VkDeviceSize>(kPackedFloatsPerParticle * sizeof(float));

enum class LayoutVariant : uint8_t {
    kStd140 = 0U,
    kStd430 = 1U,
    kPacked = 2U,
};

struct LogicalParticle {
    std::array<float, 3> coeffs{};
    std::array<float, 3> position{};
    float mass = 0.0F;
    std::array<float, 3> velocity{};
    float dt = 0.0F;
    std::array<float, 4> color{};
    float scalar = 0.0F;
};

struct alignas(16) ParticleStd430 {
    std::array<float, 3> coeffs{};
    float coeffs_padding_0 = 0.0F;
    std::array<float, 3> position{};
    float mass = 0.0F;
    std::array<float, 3> velocity{};
    float dt = 0.0F;
    std::array<float, 4> color{};
    float scalar = 0.0F;
    std::array<float, 3> tail_padding{};
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleStd430);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleStd430);
GPU_LAYOUT_ASSERT_SIZE(ParticleStd430, 80);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleStd430, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, coeffs, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, position, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, mass, 28);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, velocity, 32);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, dt, 44);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, color, 48);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd430, scalar, 64);

struct alignas(16) ParticleStd140 {
    float coeffs0 = 0.0F;
    std::array<float, 3> coeffs0_padding{};
    float coeffs1 = 0.0F;
    std::array<float, 3> coeffs1_padding{};
    float coeffs2 = 0.0F;
    std::array<float, 3> coeffs2_padding{};
    std::array<float, 3> position{};
    float mass = 0.0F;
    std::array<float, 3> velocity{};
    float dt = 0.0F;
    std::array<float, 4> color{};
    float scalar = 0.0F;
    std::array<float, 3> tail_padding{};
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleStd140);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleStd140);
GPU_LAYOUT_ASSERT_SIZE(ParticleStd140, 112);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleStd140, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, coeffs0, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, coeffs1, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, coeffs2, 32);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, position, 48);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, mass, 60);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, velocity, 64);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, dt, 76);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, color, 80);
GPU_LAYOUT_ASSERT_OFFSET(ParticleStd140, scalar, 96);

constexpr VkDeviceSize kStd430StorageStrideBytes = sizeof(ParticleStd430);
constexpr VkDeviceSize kStd140StorageStrideBytes = sizeof(ParticleStd140);
constexpr VkDeviceSize kLogicalPayloadBytes = kPackedStorageStrideBytes;

constexpr uint32_t kPackedCoeff0Index = 0U;
constexpr uint32_t kPackedCoeff1Index = 1U;
constexpr uint32_t kPackedCoeff2Index = 2U;
constexpr uint32_t kPackedPosXIndex = 3U;
constexpr uint32_t kPackedPosYIndex = 4U;
constexpr uint32_t kPackedPosZIndex = 5U;
constexpr uint32_t kPackedMassIndex = 6U;
constexpr uint32_t kPackedVelXIndex = 7U;
constexpr uint32_t kPackedVelYIndex = 8U;
constexpr uint32_t kPackedVelZIndex = 9U;
constexpr uint32_t kPackedDtIndex = 10U;
constexpr uint32_t kPackedColorRIndex = 11U;
constexpr uint32_t kPackedColorGIndex = 12U;
constexpr uint32_t kPackedColorBIndex = 13U;
constexpr uint32_t kPackedColorAIndex = 14U;
constexpr uint32_t kPackedScalarIndex = 15U;

struct SingleBufferPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    BufferResource storage_buffer{};
    void* mapped_ptr = nullptr;
};

std::vector<uint32_t> make_particle_counts(uint32_t max_particles) {
    const std::array<uint32_t, 5> base_counts = {131072U, 262144U, 524288U, 1048576U, 2097152U};
    std::vector<uint32_t> output;
    output.reserve(base_counts.size() + 4U);

    for (uint32_t value : base_counts) {
        if (value <= max_particles) {
            output.push_back(value);
        }
    }

    if (output.empty()) {
        const std::array<uint32_t, 4> fallback_counts = {4096U, 8192U, 16384U, 32768U};
        for (uint32_t value : fallback_counts) {
            if (value <= max_particles) {
                output.push_back(value);
            }
        }
    }

    if (output.empty() && max_particles > 0U) {
        output.push_back(max_particles);
    }

    std::ranges::sort(output);
    output.erase(std::ranges::unique(output).begin(), output.end());
    return output;
}

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

bool nearly_equal(float lhs, float rhs) {
    const float scale = std::max({1.0F, std::fabs(lhs), std::fabs(rhs)});
    return std::fabs(lhs - rhs) <= (kValidationEpsilon * scale);
}

LogicalParticle make_seed_particle(uint32_t index) {
    const int32_t centered_x = static_cast<int32_t>(index % 257U) - 128;
    const int32_t centered_y = static_cast<int32_t>(index % 131U) - 65;
    const int32_t centered_z = static_cast<int32_t>(index % 521U) - 260;
    const int32_t centered_vx = static_cast<int32_t>(index % 71U) - 35;
    const int32_t centered_vy = static_cast<int32_t>(index % 67U) - 33;
    const int32_t centered_vz = static_cast<int32_t>(index % 73U) - 36;

    LogicalParticle particle{};
    particle.coeffs[0] = 0.50F + (static_cast<float>(index % 17U) * 0.01F);
    particle.coeffs[1] = -0.75F + (static_cast<float>(index % 19U) * 0.02F);
    particle.coeffs[2] = 1.00F + (static_cast<float>(index % 23U) * 0.005F);
    particle.position[0] = static_cast<float>(centered_x) * 0.125F;
    particle.position[1] = static_cast<float>(centered_y) * 0.25F;
    particle.position[2] = static_cast<float>(centered_z) * 0.0625F;
    particle.mass = 1.0F + (static_cast<float>(index % 29U) * 0.02F);
    particle.velocity[0] = static_cast<float>(centered_vx) * 0.015F;
    particle.velocity[1] = static_cast<float>(centered_vy) * 0.02F;
    particle.velocity[2] = static_cast<float>(centered_vz) * 0.01F;
    particle.dt = 0.004F + (static_cast<float>(index % 13U) * 0.0005F);
    particle.color[0] = static_cast<float>(index % 97U) / 96.0F;
    particle.color[1] = static_cast<float>((index * 3U) % 89U) / 88.0F;
    particle.color[2] = static_cast<float>((index * 5U) % 83U) / 82.0F;
    particle.color[3] = 0.25F + (static_cast<float>(index % 31U) * 0.01F);
    particle.scalar = -0.50F + (static_cast<float>(index % 37U) * 0.03F);
    return particle;
}

void advance_particle(LogicalParticle& particle) {
    const float adv_x = (particle.velocity[0] * particle.dt) + (particle.coeffs[0] * 0.001F);
    const float adv_y = (particle.velocity[1] * particle.dt) + (particle.coeffs[1] * 0.001F);
    const float adv_z = (particle.velocity[2] * particle.dt) + (particle.coeffs[2] * 0.001F);

    particle.position[0] += adv_x;
    particle.position[1] += adv_y;
    particle.position[2] += adv_z;
    particle.mass += (particle.scalar * 0.0001F) + 0.000001F;
    particle.velocity[0] += particle.coeffs[0] * 0.0002F;
    particle.velocity[1] += particle.coeffs[1] * 0.0002F;
    particle.velocity[2] += particle.coeffs[2] * 0.0002F;
    particle.color[0] = (particle.color[0] * 0.999F) + (adv_x * 0.001F);
    particle.color[1] = (particle.color[1] * 0.999F) + (adv_y * 0.001F);
    particle.color[2] = (particle.color[2] * 0.999F) + (adv_z * 0.001F);
    particle.color[3] += 0.0005F;
    particle.scalar = (particle.scalar * 0.5F) + (particle.mass * 0.01F);
    particle.coeffs[0] += 0.0001F;
    particle.coeffs[1] += 0.0002F;
    particle.coeffs[2] += 0.0003F;
}

LogicalParticle make_expected_particle(uint32_t index) {
    LogicalParticle expected = make_seed_particle(index);
    advance_particle(expected);
    return expected;
}

bool validate_particle(const LogicalParticle& actual, const LogicalParticle& expected) {
    for (uint32_t i = 0U; i < actual.coeffs.size(); ++i) {
        if (!nearly_equal(actual.coeffs[i], expected.coeffs[i])) {
            return false;
        }
    }

    for (uint32_t i = 0U; i < actual.position.size(); ++i) {
        if (!nearly_equal(actual.position[i], expected.position[i])) {
            return false;
        }
    }

    if (!nearly_equal(actual.mass, expected.mass)) {
        return false;
    }

    for (uint32_t i = 0U; i < actual.velocity.size(); ++i) {
        if (!nearly_equal(actual.velocity[i], expected.velocity[i])) {
            return false;
        }
    }

    if (!nearly_equal(actual.dt, expected.dt)) {
        return false;
    }

    for (uint32_t i = 0U; i < actual.color.size(); ++i) {
        if (!nearly_equal(actual.color[i], expected.color[i])) {
            return false;
        }
    }

    return nearly_equal(actual.scalar, expected.scalar);
}

void pack_particle_std430(const LogicalParticle& input, ParticleStd430& output) {
    output.coeffs[0] = input.coeffs[0];
    output.coeffs[1] = input.coeffs[1];
    output.coeffs[2] = input.coeffs[2];
    output.coeffs_padding_0 = 0.0F;
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.mass = input.mass;
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.dt = input.dt;
    output.color[0] = input.color[0];
    output.color[1] = input.color[1];
    output.color[2] = input.color[2];
    output.color[3] = input.color[3];
    output.scalar = input.scalar;
    output.tail_padding[0] = 0.0F;
    output.tail_padding[1] = 0.0F;
    output.tail_padding[2] = 0.0F;
}

LogicalParticle unpack_particle_std430(const ParticleStd430& input) {
    LogicalParticle output{};
    output.coeffs[0] = input.coeffs[0];
    output.coeffs[1] = input.coeffs[1];
    output.coeffs[2] = input.coeffs[2];
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.mass = input.mass;
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.dt = input.dt;
    output.color[0] = input.color[0];
    output.color[1] = input.color[1];
    output.color[2] = input.color[2];
    output.color[3] = input.color[3];
    output.scalar = input.scalar;
    return output;
}

void pack_particle_std140(const LogicalParticle& input, ParticleStd140& output) {
    output.coeffs0 = input.coeffs[0];
    output.coeffs0_padding[0] = 0.0F;
    output.coeffs0_padding[1] = 0.0F;
    output.coeffs0_padding[2] = 0.0F;
    output.coeffs1 = input.coeffs[1];
    output.coeffs1_padding[0] = 0.0F;
    output.coeffs1_padding[1] = 0.0F;
    output.coeffs1_padding[2] = 0.0F;
    output.coeffs2 = input.coeffs[2];
    output.coeffs2_padding[0] = 0.0F;
    output.coeffs2_padding[1] = 0.0F;
    output.coeffs2_padding[2] = 0.0F;
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.mass = input.mass;
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.dt = input.dt;
    output.color[0] = input.color[0];
    output.color[1] = input.color[1];
    output.color[2] = input.color[2];
    output.color[3] = input.color[3];
    output.scalar = input.scalar;
    output.tail_padding[0] = 0.0F;
    output.tail_padding[1] = 0.0F;
    output.tail_padding[2] = 0.0F;
}

LogicalParticle unpack_particle_std140(const ParticleStd140& input) {
    LogicalParticle output{};
    output.coeffs[0] = input.coeffs0;
    output.coeffs[1] = input.coeffs1;
    output.coeffs[2] = input.coeffs2;
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.mass = input.mass;
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.dt = input.dt;
    output.color[0] = input.color[0];
    output.color[1] = input.color[1];
    output.color[2] = input.color[2];
    output.color[3] = input.color[3];
    output.scalar = input.scalar;
    return output;
}

float* packed_particle_ptr(float* data, uint32_t index) {
    return data + (static_cast<std::size_t>(index) * kPackedFloatsPerParticle);
}

const float* packed_particle_ptr(const float* data, uint32_t index) {
    return data + (static_cast<std::size_t>(index) * kPackedFloatsPerParticle);
}

void pack_particle_packed(const LogicalParticle& input, float* output) {
    output[kPackedCoeff0Index] = input.coeffs[0];
    output[kPackedCoeff1Index] = input.coeffs[1];
    output[kPackedCoeff2Index] = input.coeffs[2];
    output[kPackedPosXIndex] = input.position[0];
    output[kPackedPosYIndex] = input.position[1];
    output[kPackedPosZIndex] = input.position[2];
    output[kPackedMassIndex] = input.mass;
    output[kPackedVelXIndex] = input.velocity[0];
    output[kPackedVelYIndex] = input.velocity[1];
    output[kPackedVelZIndex] = input.velocity[2];
    output[kPackedDtIndex] = input.dt;
    output[kPackedColorRIndex] = input.color[0];
    output[kPackedColorGIndex] = input.color[1];
    output[kPackedColorBIndex] = input.color[2];
    output[kPackedColorAIndex] = input.color[3];
    output[kPackedScalarIndex] = input.scalar;
}

LogicalParticle unpack_particle_packed(const float* input) {
    LogicalParticle output{};
    output.coeffs[0] = input[kPackedCoeff0Index];
    output.coeffs[1] = input[kPackedCoeff1Index];
    output.coeffs[2] = input[kPackedCoeff2Index];
    output.position[0] = input[kPackedPosXIndex];
    output.position[1] = input[kPackedPosYIndex];
    output.position[2] = input[kPackedPosZIndex];
    output.mass = input[kPackedMassIndex];
    output.velocity[0] = input[kPackedVelXIndex];
    output.velocity[1] = input[kPackedVelYIndex];
    output.velocity[2] = input[kPackedVelZIndex];
    output.dt = input[kPackedDtIndex];
    output.color[0] = input[kPackedColorRIndex];
    output.color[1] = input[kPackedColorGIndex];
    output.color[2] = input[kPackedColorBIndex];
    output.color[3] = input[kPackedColorAIndex];
    output.scalar = input[kPackedScalarIndex];
    return output;
}

void fill_std430_seed_particles(ParticleStd430* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        pack_particle_std430(make_seed_particle(index), particles[index]);
    }
}

void fill_std140_seed_particles(ParticleStd140* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        pack_particle_std140(make_seed_particle(index), particles[index]);
    }
}

void fill_packed_seed_particles(float* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        pack_particle_packed(make_seed_particle(index), packed_particle_ptr(particles, index));
    }
}

bool validate_std430_particles(const ParticleStd430* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = unpack_particle_std430(particles[index]);
        if (!validate_particle(actual, expected)) {
            return false;
        }
    }

    return true;
}

bool validate_std140_particles(const ParticleStd140* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = unpack_particle_std140(particles[index]);
        if (!validate_particle(actual, expected)) {
            return false;
        }
    }

    return true;
}

bool validate_packed_particles(const float* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = unpack_particle_packed(packed_particle_ptr(particles, index));
        if (!validate_particle(actual, expected)) {
            return false;
        }
    }

    return true;
}

bool create_single_buffer_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                             VkDeviceSize buffer_size, const char* label,
                                             SingleBufferPipelineResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.storage_buffer)) {
        std::cerr << "Failed to create storage buffer for " << label << ".\n";
        return false;
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load shader module for " << label << ": " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create descriptor set layout for " << label << ".\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create descriptor pool for " << label << ".\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate descriptor set for " << label << ".\n";
        return false;
    }

    const VkDescriptorBufferInfo buffer_info{
        out_resources.storage_buffer.buffer,
        0U,
        out_resources.storage_buffer.size,
    };
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), out_resources.descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = buffer_info,
                                                          },
                                                      });

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{
            VK_SHADER_STAGE_COMPUTE_BIT,
            0U,
            sizeof(uint32_t),
        },
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create pipeline layout for " << label << ".\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create compute pipeline for " << label << ".\n";
        return false;
    }

    if (!map_buffer_memory(context, out_resources.storage_buffer, label, out_resources.mapped_ptr)) {
        return false;
    }

    return true;
}

void cleanup_single_buffer_resources(VulkanContext& context, SingleBufferPipelineResources& resources) {
    if (resources.mapped_ptr != nullptr && resources.storage_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.storage_buffer.memory);
        resources.mapped_ptr = nullptr;
    }

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
    destroy_buffer_resource(context.device(), resources.storage_buffer);
}

double run_dispatch(VulkanContext& context, uint32_t particles, const SingleBufferPipelineResources& resources) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);
    if (group_count == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(particles), &particles);
        for (uint32_t dispatch_index = 0U; dispatch_index < kDispatchCount; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count, 1U, 1U);
        }
    });
}

VkDeviceSize storage_stride_bytes(LayoutVariant variant) {
    if (variant == LayoutVariant::kStd140) {
        return kStd140StorageStrideBytes;
    }
    if (variant == LayoutVariant::kStd430) {
        return kStd430StorageStrideBytes;
    }
    return kPackedStorageStrideBytes;
}

double measured_bytes_per_particle() {
    return static_cast<double>(kLogicalPayloadBytes * 2U);
}

double alignment_waste_ratio(LayoutVariant variant) {
    const auto stride_bytes = static_cast<double>(storage_stride_bytes(variant));
    const auto logical_bytes = static_cast<double>(kLogicalPayloadBytes);
    return (stride_bytes - logical_bytes) / logical_bytes;
}

double compute_effective_gbps(uint32_t particles, uint32_t dispatch_count, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes =
        measured_bytes_per_particle() * static_cast<double>(particles) * static_cast<double>(dispatch_count);
    return bytes / (dispatch_gpu_ms * 1.0e6);
}

std::string build_case_name(const std::string& variant, uint32_t particles) {
    return std::string(kExperimentId) + "_" + variant + "_particles_" + std::to_string(particles);
}

template <typename PrepareFn, typename ValidateFn>
void run_variant_case(VulkanContext& context, const BenchmarkRunner& runner, LayoutVariant layout_variant,
                      const std::string& variant_name, uint32_t particles,
                      const SingleBufferPipelineResources& resources, PrepareFn prepare_inputs,
                      ValidateFn validate_outputs, bool verbose_progress, Std430Std140PackedExperimentOutput& output) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant start: name=" << variant_name << ", particles=" << particles
                  << ", group_count_x=" << group_count << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        prepare_inputs(particles);
        const double dispatch_ms = run_dispatch(context, particles, resources);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(particles);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", particles=" << particles << ", gpu_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] warmup issue for variant=" << variant_name
                      << ", particles=" << particles << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        prepare_inputs(particles);
        const double dispatch_ms = run_dispatch(context, particles, resources);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(particles);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        std::string notes;
        append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
        append_note(notes, "group_count_x=" + std::to_string(group_count));
        append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
        append_note(notes, "storage_bytes_per_particle=" +
                               std::to_string(static_cast<uint64_t>(storage_stride_bytes(layout_variant))));
        append_note(notes, "logical_bytes_per_particle=" + std::to_string(static_cast<uint64_t>(kLogicalPayloadBytes)));
        append_note(notes, "alignment_waste_ratio=" + std::to_string(alignment_waste_ratio(layout_variant)));
        if (!dispatch_ok) {
            append_note(notes, "dispatch_ms_non_finite");
        }
        if (!data_ok) {
            append_note(notes, "correctness_mismatch");
        }

        const bool correctness = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness;
        dispatch_samples.push_back(dispatch_ms);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", particles=" << particles << ", gpu_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = particles,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(particles, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(particles, kDispatchCount, dispatch_ms),
            .correctness_pass = correctness,
            .notes = notes,
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(build_case_name(variant_name, particles), dispatch_samples);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant complete: name=" << variant_name << ", particles=" << particles
                  << ", samples=" << summary.sample_count << ", median_gpu_ms=" << summary.median_ms << "\n";
    }
    output.summary_results.push_back(summary);
}

} // namespace

Std430Std140PackedExperimentOutput
run_std430_std140_packed_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const Std430Std140PackedExperimentConfig& config) {
    Std430Std140PackedExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "std430/std140/packed experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string std140_shader =
        VulkanComputeUtils::resolve_shader_path(config.std140_shader_path, "08_std140.comp.spv");
    const std::string std430_shader =
        VulkanComputeUtils::resolve_shader_path(config.std430_shader_path, "08_std430.comp.spv");
    const std::string packed_shader =
        VulkanComputeUtils::resolve_shader_path(config.packed_shader_path, "08_packed.comp.spv");
    if (std140_shader.empty() || std430_shader.empty() || packed_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for std430/std140/packed experiment.\n";
        return output;
    }

    const uint64_t max_particles_u64 = config.max_buffer_bytes / static_cast<uint64_t>(kStd140StorageStrideBytes);
    if (max_particles_u64 == 0U) {
        std::cerr << "Scratch buffer too small for std430/std140/packed experiment.\n";
        return output;
    }

    const uint32_t max_particles =
        static_cast<uint32_t>(std::min<uint64_t>(max_particles_u64, std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> particle_counts = make_particle_counts(max_particles);
    if (particle_counts.empty()) {
        std::cerr << "No legal particle counts available for std430/std140/packed experiment.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader std140: " << std140_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader std430: " << std430_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader packed: " << packed_shader << "\n";
        std::cout << "[" << kExperimentId << "] Storage bytes per particle: std140=" << kStd140StorageStrideBytes
                  << ", std430=" << kStd430StorageStrideBytes << ", packed=" << kPackedStorageStrideBytes
                  << ", logical=" << kLogicalPayloadBytes << "\n";
        std::cout << "[" << kExperimentId << "] Starting run with particles=" << particle_counts.size()
                  << ", variants=3, local_size_x=" << kWorkgroupSize
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    SingleBufferPipelineResources std140_resources{};
    const VkDeviceSize std140_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kStd140StorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, std140_shader, std140_buffer_size, "std140 buffer",
                                                 std140_resources)) {
        cleanup_single_buffer_resources(context, std140_resources);
        return output;
    }

    SingleBufferPipelineResources std430_resources{};
    const VkDeviceSize std430_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kStd430StorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, std430_shader, std430_buffer_size, "std430 buffer",
                                                 std430_resources)) {
        cleanup_single_buffer_resources(context, std140_resources);
        cleanup_single_buffer_resources(context, std430_resources);
        return output;
    }

    SingleBufferPipelineResources packed_resources{};
    const VkDeviceSize packed_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kPackedStorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, packed_shader, packed_buffer_size, "packed buffer",
                                                 packed_resources)) {
        cleanup_single_buffer_resources(context, std140_resources);
        cleanup_single_buffer_resources(context, std430_resources);
        cleanup_single_buffer_resources(context, packed_resources);
        return output;
    }

    auto* std140_particles = static_cast<ParticleStd140*>(std140_resources.mapped_ptr);
    auto* std430_particles = static_cast<ParticleStd430*>(std430_resources.mapped_ptr);
    auto* packed_particles = static_cast<float*>(packed_resources.mapped_ptr);
    if (std140_particles == nullptr || std430_particles == nullptr || packed_particles == nullptr) {
        std::cerr << "Mapped pointers are missing for std430/std140/packed experiment resources.\n";
        cleanup_single_buffer_resources(context, std140_resources);
        cleanup_single_buffer_resources(context, std430_resources);
        cleanup_single_buffer_resources(context, packed_resources);
        return output;
    }

    for (uint32_t particles : particle_counts) {
        run_variant_case(
            context, runner, LayoutVariant::kStd140, "std140", particles, std140_resources,
            [&](uint32_t count) {
                fill_std140_seed_particles(std140_particles, count);
            },
            [&](uint32_t count) {
                return validate_std140_particles(std140_particles, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, LayoutVariant::kStd430, "std430", particles, std430_resources,
            [&](uint32_t count) {
                fill_std430_seed_particles(std430_particles, count);
            },
            [&](uint32_t count) {
                return validate_std430_particles(std430_particles, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, LayoutVariant::kPacked, "packed", particles, packed_resources,
            [&](uint32_t count) {
                fill_packed_seed_particles(packed_particles, count);
            },
            [&](uint32_t count) {
                return validate_packed_particles(packed_particles, count);
            },
            verbose_progress, output);
    }

    cleanup_single_buffer_resources(context, std140_resources);
    cleanup_single_buffer_resources(context, std430_resources);
    cleanup_single_buffer_resources(context, packed_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
