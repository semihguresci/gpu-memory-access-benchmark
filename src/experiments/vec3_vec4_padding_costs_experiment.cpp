#include "experiments/vec3_vec4_padding_costs_experiment.hpp"

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

constexpr const char* kExperimentId = "09_vec3_vec4_padding_costs";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr float kValidationEpsilon = 1.0e-5F;
constexpr uint32_t kSplitScalarsPerParticle = 11U;
constexpr VkDeviceSize kLogicalPayloadBytes = static_cast<VkDeviceSize>(kSplitScalarsPerParticle * sizeof(float));

enum class LayoutVariant : uint8_t {
    kVec3Padded = 0U,
    kVec4 = 1U,
    kSplitScalars = 2U,
};

struct LogicalParticle {
    std::array<float, 3> coeffs{};
    std::array<float, 3> position{};
    std::array<float, 3> velocity{};
    float mass = 0.0F;
    float dt = 0.0F;
};

struct alignas(16) ParticleVec3Padded {
    std::array<float, 3> coeffs{};
    float coeffs_padding_0 = 0.0F;
    std::array<float, 3> position{};
    float position_padding_0 = 0.0F;
    std::array<float, 3> velocity{};
    float mass = 0.0F;
    float dt = 0.0F;
    std::array<float, 3> tail_padding{};
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleVec3Padded);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleVec3Padded);
GPU_LAYOUT_ASSERT_SIZE(ParticleVec3Padded, 64);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleVec3Padded, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec3Padded, coeffs, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec3Padded, position, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec3Padded, velocity, 32);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec3Padded, mass, 44);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec3Padded, dt, 48);

struct alignas(16) ParticleVec4 {
    std::array<float, 4> coeffs_mass{};
    std::array<float, 4> position_dt{};
    std::array<float, 4> velocity_padding{};
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleVec4);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleVec4);
GPU_LAYOUT_ASSERT_SIZE(ParticleVec4, 48);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleVec4, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec4, coeffs_mass, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec4, position_dt, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleVec4, velocity_padding, 32);

constexpr VkDeviceSize kVec3PaddedStorageStrideBytes = sizeof(ParticleVec3Padded);
constexpr VkDeviceSize kVec4StorageStrideBytes = sizeof(ParticleVec4);
constexpr VkDeviceSize kSplitScalarsStorageStrideBytes = kLogicalPayloadBytes;

constexpr uint32_t kSplitCoeff0Array = 0U;
constexpr uint32_t kSplitCoeff1Array = 1U;
constexpr uint32_t kSplitCoeff2Array = 2U;
constexpr uint32_t kSplitPosXArray = 3U;
constexpr uint32_t kSplitPosYArray = 4U;
constexpr uint32_t kSplitPosZArray = 5U;
constexpr uint32_t kSplitVelXArray = 6U;
constexpr uint32_t kSplitVelYArray = 7U;
constexpr uint32_t kSplitVelZArray = 8U;
constexpr uint32_t kSplitMassArray = 9U;
constexpr uint32_t kSplitDtArray = 10U;

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

    for (const uint32_t value : base_counts) {
        if (value <= max_particles) {
            output.push_back(value);
        }
    }

    if (output.empty()) {
        const std::array<uint32_t, 4> fallback_counts = {4096U, 8192U, 16384U, 32768U};
        for (const uint32_t value : fallback_counts) {
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
    particle.coeffs[0] = 0.45F + (static_cast<float>(index % 17U) * 0.02F);
    particle.coeffs[1] = -0.65F + (static_cast<float>(index % 19U) * 0.015F);
    particle.coeffs[2] = 1.10F + (static_cast<float>(index % 23U) * 0.01F);
    particle.position[0] = static_cast<float>(centered_x) * 0.125F;
    particle.position[1] = static_cast<float>(centered_y) * 0.25F;
    particle.position[2] = static_cast<float>(centered_z) * 0.0625F;
    particle.velocity[0] = static_cast<float>(centered_vx) * 0.01F;
    particle.velocity[1] = static_cast<float>(centered_vy) * 0.015F;
    particle.velocity[2] = static_cast<float>(centered_vz) * 0.02F;
    particle.mass = 1.0F + (static_cast<float>(index % 31U) * 0.05F);
    particle.dt = 0.004F + (static_cast<float>(index % 13U) * 0.00025F);
    return particle;
}

void advance_particle(LogicalParticle& particle) {
    const float adv_x = (particle.velocity[0] * particle.dt) + (particle.coeffs[0] * 0.001F);
    const float adv_y = (particle.velocity[1] * particle.dt) + (particle.coeffs[1] * 0.001F);
    const float adv_z = (particle.velocity[2] * particle.dt) + (particle.coeffs[2] * 0.001F);

    particle.position[0] += adv_x;
    particle.position[1] += adv_y;
    particle.position[2] += adv_z;
    particle.velocity[0] += particle.coeffs[0] * 0.0002F;
    particle.velocity[1] += particle.coeffs[1] * 0.0002F;
    particle.velocity[2] += particle.coeffs[2] * 0.0002F;
    particle.mass += (particle.coeffs[0] * 0.0003F) + (particle.position[2] * 0.00001F);
    particle.dt = (particle.dt * 0.999F) + 0.0000025F;
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
    for (std::size_t i = 0U; i < actual.coeffs.size(); ++i) {
        if (!nearly_equal(actual.coeffs[i], expected.coeffs[i])) {
            return false;
        }
    }

    for (std::size_t i = 0U; i < actual.position.size(); ++i) {
        if (!nearly_equal(actual.position[i], expected.position[i])) {
            return false;
        }
    }

    for (std::size_t i = 0U; i < actual.velocity.size(); ++i) {
        if (!nearly_equal(actual.velocity[i], expected.velocity[i])) {
            return false;
        }
    }

    if (!nearly_equal(actual.mass, expected.mass)) {
        return false;
    }

    return nearly_equal(actual.dt, expected.dt);
}

void pack_particle_vec3_padded(const LogicalParticle& input, ParticleVec3Padded& output) {
    output.coeffs[0] = input.coeffs[0];
    output.coeffs[1] = input.coeffs[1];
    output.coeffs[2] = input.coeffs[2];
    output.coeffs_padding_0 = 0.0F;
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.position_padding_0 = 0.0F;
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.mass = input.mass;
    output.dt = input.dt;
    output.tail_padding[0] = 0.0F;
    output.tail_padding[1] = 0.0F;
    output.tail_padding[2] = 0.0F;
}

LogicalParticle unpack_particle_vec3_padded(const ParticleVec3Padded& input) {
    LogicalParticle output{};
    output.coeffs[0] = input.coeffs[0];
    output.coeffs[1] = input.coeffs[1];
    output.coeffs[2] = input.coeffs[2];
    output.position[0] = input.position[0];
    output.position[1] = input.position[1];
    output.position[2] = input.position[2];
    output.velocity[0] = input.velocity[0];
    output.velocity[1] = input.velocity[1];
    output.velocity[2] = input.velocity[2];
    output.mass = input.mass;
    output.dt = input.dt;
    return output;
}

void pack_particle_vec4(const LogicalParticle& input, ParticleVec4& output) {
    output.coeffs_mass[0] = input.coeffs[0];
    output.coeffs_mass[1] = input.coeffs[1];
    output.coeffs_mass[2] = input.coeffs[2];
    output.coeffs_mass[3] = input.mass;
    output.position_dt[0] = input.position[0];
    output.position_dt[1] = input.position[1];
    output.position_dt[2] = input.position[2];
    output.position_dt[3] = input.dt;
    output.velocity_padding[0] = input.velocity[0];
    output.velocity_padding[1] = input.velocity[1];
    output.velocity_padding[2] = input.velocity[2];
    output.velocity_padding[3] = 0.0F;
}

LogicalParticle unpack_particle_vec4(const ParticleVec4& input) {
    LogicalParticle output{};
    output.coeffs[0] = input.coeffs_mass[0];
    output.coeffs[1] = input.coeffs_mass[1];
    output.coeffs[2] = input.coeffs_mass[2];
    output.position[0] = input.position_dt[0];
    output.position[1] = input.position_dt[1];
    output.position[2] = input.position_dt[2];
    output.velocity[0] = input.velocity_padding[0];
    output.velocity[1] = input.velocity_padding[1];
    output.velocity[2] = input.velocity_padding[2];
    output.mass = input.coeffs_mass[3];
    output.dt = input.position_dt[3];
    return output;
}

float* split_array_base(float* data, uint32_t particle_count, uint32_t array_index) {
    return data + (static_cast<std::size_t>(particle_count) * static_cast<std::size_t>(array_index));
}

const float* split_array_base(const float* data, uint32_t particle_count, uint32_t array_index) {
    return data + (static_cast<std::size_t>(particle_count) * static_cast<std::size_t>(array_index));
}

void set_split_particle(float* data, uint32_t particle_count, uint32_t particle_index, const LogicalParticle& input) {
    split_array_base(data, particle_count, kSplitCoeff0Array)[particle_index] = input.coeffs[0];
    split_array_base(data, particle_count, kSplitCoeff1Array)[particle_index] = input.coeffs[1];
    split_array_base(data, particle_count, kSplitCoeff2Array)[particle_index] = input.coeffs[2];
    split_array_base(data, particle_count, kSplitPosXArray)[particle_index] = input.position[0];
    split_array_base(data, particle_count, kSplitPosYArray)[particle_index] = input.position[1];
    split_array_base(data, particle_count, kSplitPosZArray)[particle_index] = input.position[2];
    split_array_base(data, particle_count, kSplitVelXArray)[particle_index] = input.velocity[0];
    split_array_base(data, particle_count, kSplitVelYArray)[particle_index] = input.velocity[1];
    split_array_base(data, particle_count, kSplitVelZArray)[particle_index] = input.velocity[2];
    split_array_base(data, particle_count, kSplitMassArray)[particle_index] = input.mass;
    split_array_base(data, particle_count, kSplitDtArray)[particle_index] = input.dt;
}

LogicalParticle get_split_particle(const float* data, uint32_t particle_count, uint32_t particle_index) {
    LogicalParticle output{};
    output.coeffs[0] = split_array_base(data, particle_count, kSplitCoeff0Array)[particle_index];
    output.coeffs[1] = split_array_base(data, particle_count, kSplitCoeff1Array)[particle_index];
    output.coeffs[2] = split_array_base(data, particle_count, kSplitCoeff2Array)[particle_index];
    output.position[0] = split_array_base(data, particle_count, kSplitPosXArray)[particle_index];
    output.position[1] = split_array_base(data, particle_count, kSplitPosYArray)[particle_index];
    output.position[2] = split_array_base(data, particle_count, kSplitPosZArray)[particle_index];
    output.velocity[0] = split_array_base(data, particle_count, kSplitVelXArray)[particle_index];
    output.velocity[1] = split_array_base(data, particle_count, kSplitVelYArray)[particle_index];
    output.velocity[2] = split_array_base(data, particle_count, kSplitVelZArray)[particle_index];
    output.mass = split_array_base(data, particle_count, kSplitMassArray)[particle_index];
    output.dt = split_array_base(data, particle_count, kSplitDtArray)[particle_index];
    return output;
}

void fill_vec3_padded_seed_particles(ParticleVec3Padded* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        pack_particle_vec3_padded(make_seed_particle(index), particles[index]);
    }
}

void fill_vec4_seed_particles(ParticleVec4* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        pack_particle_vec4(make_seed_particle(index), particles[index]);
    }
}

void fill_split_seed_particles(float* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        set_split_particle(particles, count, index, make_seed_particle(index));
    }
}

bool validate_vec3_padded_particles(const ParticleVec3Padded* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = unpack_particle_vec3_padded(particles[index]);
        if (!validate_particle(actual, expected)) {
            return false;
        }
    }

    return true;
}

bool validate_vec4_particles(const ParticleVec4* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = unpack_particle_vec4(particles[index]);
        if (!validate_particle(actual, expected)) {
            return false;
        }
    }

    return true;
}

bool validate_split_particles(const float* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const LogicalParticle expected = make_expected_particle(index);
        const LogicalParticle actual = get_split_particle(particles, count, index);
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
    if (variant == LayoutVariant::kVec3Padded) {
        return kVec3PaddedStorageStrideBytes;
    }
    if (variant == LayoutVariant::kVec4) {
        return kVec4StorageStrideBytes;
    }
    return kSplitScalarsStorageStrideBytes;
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
                      ValidateFn validate_outputs, bool verbose_progress,
                      Vec3Vec4PaddingCostsExperimentOutput& output) {
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

Vec3Vec4PaddingCostsExperimentOutput
run_vec3_vec4_padding_costs_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const Vec3Vec4PaddingCostsExperimentConfig& config) {
    Vec3Vec4PaddingCostsExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "vec3/vec4 padding cost experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string vec3_shader =
        VulkanComputeUtils::resolve_shader_path(config.vec3_shader_path, "09_vec3_padded.comp.spv");
    const std::string vec4_shader =
        VulkanComputeUtils::resolve_shader_path(config.vec4_shader_path, "09_vec4.comp.spv");
    const std::string split_shader =
        VulkanComputeUtils::resolve_shader_path(config.split_scalars_shader_path, "09_split_scalars.comp.spv");
    if (vec3_shader.empty() || vec4_shader.empty() || split_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for vec3/vec4 padding cost experiment.\n";
        return output;
    }

    const uint64_t max_particles_u64 = config.max_buffer_bytes / static_cast<uint64_t>(kVec3PaddedStorageStrideBytes);
    if (max_particles_u64 == 0U) {
        std::cerr << "Scratch buffer too small for vec3/vec4 padding cost experiment.\n";
        return output;
    }

    const uint32_t max_particles =
        static_cast<uint32_t>(std::min<uint64_t>(max_particles_u64, std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> particle_counts = make_particle_counts(max_particles);
    if (particle_counts.empty()) {
        std::cerr << "No legal particle counts available for vec3/vec4 padding cost experiment.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader vec3_padded: " << vec3_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader vec4: " << vec4_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader split_scalars: " << split_shader << "\n";
        std::cout << "[" << kExperimentId
                  << "] Storage bytes per particle: vec3_padded=" << kVec3PaddedStorageStrideBytes
                  << ", vec4=" << kVec4StorageStrideBytes << ", split_scalars=" << kSplitScalarsStorageStrideBytes
                  << ", logical=" << kLogicalPayloadBytes << "\n";
        std::cout << "[" << kExperimentId << "] Starting run with particles=" << particle_counts.size()
                  << ", variants=3, local_size_x=" << kWorkgroupSize
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    SingleBufferPipelineResources vec3_resources{};
    const VkDeviceSize vec3_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kVec3PaddedStorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, vec3_shader, vec3_buffer_size, "vec3_padded buffer",
                                                 vec3_resources)) {
        cleanup_single_buffer_resources(context, vec3_resources);
        return output;
    }

    SingleBufferPipelineResources vec4_resources{};
    const VkDeviceSize vec4_buffer_size = static_cast<VkDeviceSize>(particle_counts.back()) * kVec4StorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, vec4_shader, vec4_buffer_size, "vec4 buffer",
                                                 vec4_resources)) {
        cleanup_single_buffer_resources(context, vec3_resources);
        cleanup_single_buffer_resources(context, vec4_resources);
        return output;
    }

    SingleBufferPipelineResources split_resources{};
    const VkDeviceSize split_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kSplitScalarsStorageStrideBytes;
    if (!create_single_buffer_pipeline_resources(context, split_shader, split_buffer_size, "split_scalars buffer",
                                                 split_resources)) {
        cleanup_single_buffer_resources(context, vec3_resources);
        cleanup_single_buffer_resources(context, vec4_resources);
        cleanup_single_buffer_resources(context, split_resources);
        return output;
    }

    auto* vec3_particles = static_cast<ParticleVec3Padded*>(vec3_resources.mapped_ptr);
    auto* vec4_particles = static_cast<ParticleVec4*>(vec4_resources.mapped_ptr);
    auto* split_particles = static_cast<float*>(split_resources.mapped_ptr);
    if (vec3_particles == nullptr || vec4_particles == nullptr || split_particles == nullptr) {
        std::cerr << "Mapped pointers are missing for vec3/vec4 padding cost experiment resources.\n";
        cleanup_single_buffer_resources(context, vec3_resources);
        cleanup_single_buffer_resources(context, vec4_resources);
        cleanup_single_buffer_resources(context, split_resources);
        return output;
    }

    for (const uint32_t particles : particle_counts) {
        run_variant_case(
            context, runner, LayoutVariant::kVec3Padded, "vec3_padded", particles, vec3_resources,
            [&](uint32_t count) {
                fill_vec3_padded_seed_particles(vec3_particles, count);
            },
            [&](uint32_t count) {
                return validate_vec3_padded_particles(vec3_particles, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, LayoutVariant::kVec4, "vec4", particles, vec4_resources,
            [&](uint32_t count) {
                fill_vec4_seed_particles(vec4_particles, count);
            },
            [&](uint32_t count) {
                return validate_vec4_particles(vec4_particles, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, LayoutVariant::kSplitScalars, "split_scalars", particles, split_resources,
            [&](uint32_t count) {
                fill_split_seed_particles(split_particles, count);
            },
            [&](uint32_t count) {
                return validate_split_particles(split_particles, count);
            },
            verbose_progress, output);
    }

    cleanup_single_buffer_resources(context, vec3_resources);
    cleanup_single_buffer_resources(context, vec4_resources);
    cleanup_single_buffer_resources(context, split_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
