#include "experiments/aosoa_blocked_layout_experiment.hpp"

#include "utils/buffer_utils.hpp"
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

constexpr const char* kExperimentId = "07_aosoa_blocked_layout";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kFloatArraysForSoa = 16U;
constexpr uint32_t kFloatsPerParticle = 16U;
constexpr uint32_t kDispatchCount = 1U;
constexpr float kMassIncrement = 0.000001F;
constexpr float kValidationEpsilon = 1.0e-5F;
constexpr std::array<uint32_t, 4> kAosoaBlockSizes = {4U, 8U, 16U, 32U};

enum class LayoutVariant : uint8_t {
    kAos = 0U,
    kSoa = 1U,
    kAosoa = 2U,
};

struct ParticleAosStd430 {
    float px;
    float py;
    float pz;
    float mass;
    float vx;
    float vy;
    float vz;
    float dt;
    float aux0;
    float aux1;
    float aux2;
    float aux3;
    float cold0;
    float cold1;
    float cold2;
    float cold3;
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleAosStd430);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleAosStd430);
GPU_LAYOUT_ASSERT_SIZE(ParticleAosStd430, 64);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleAosStd430, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, px, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, py, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, pz, 8);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, mass, 12);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vx, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vy, 20);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vz, 24);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, dt, 28);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, aux0, 32);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, aux1, 36);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, aux2, 40);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, aux3, 44);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, cold0, 48);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, cold1, 52);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, cold2, 56);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, cold3, 60);

struct AosoaPushConstants {
    uint32_t count = 0U;
    uint32_t block_size = 0U;
    uint32_t block_count = 0U;
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(AosoaPushConstants);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(AosoaPushConstants);
GPU_LAYOUT_ASSERT_SIZE(AosoaPushConstants, 12);
GPU_LAYOUT_ASSERT_ALIGNMENT(AosoaPushConstants, 4);

struct AosPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    BufferResource storage_buffer{};
    void* mapped_ptr = nullptr;
};

struct SoaPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<BufferResource> storage_buffers;
    std::array<void*, kFloatArraysForSoa> mapped_ptrs{};
};

struct AosoaPipelineResources {
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
    const std::array<uint32_t, 6> base_counts = {1000000U, 2000000U, 4000000U, 8000000U, 16000000U, 32000000U};
    std::vector<uint32_t> output;
    output.reserve(base_counts.size() + 4U);

    for (uint32_t value : base_counts) {
        if (value <= max_particles) {
            output.push_back(value);
        }
    }

    if (output.empty()) {
        const std::array<uint32_t, 4> fallback_counts = {131072U, 262144U, 524288U, 1048576U};
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

uint32_t compute_block_count(uint32_t count, uint32_t block_size) {
    if (count == 0U || block_size == 0U) {
        return 0U;
    }

    return (count + block_size - 1U) / block_size;
}

uint32_t compute_aosoa_block_stride(uint32_t block_size) {
    return kFloatsPerParticle * block_size;
}

uint32_t compute_aosoa_index(uint32_t field, uint32_t particle, uint32_t block_size) {
    const uint32_t block = particle / block_size;
    const uint32_t lane = particle % block_size;
    return (block * compute_aosoa_block_stride(block_size)) + (field * block_size) + lane;
}

VkDeviceSize compute_aosoa_storage_bytes_for_particles(uint32_t particles, uint32_t block_size) {
    const auto block_count = static_cast<uint64_t>(compute_block_count(particles, block_size));
    const uint64_t total_floats = block_count * static_cast<uint64_t>(compute_aosoa_block_stride(block_size));
    return static_cast<VkDeviceSize>(total_floats * sizeof(float));
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

ParticleAosStd430 make_seed_particle(uint32_t index) {
    const int32_t centered_x = static_cast<int32_t>(index % 257U) - 128;
    const int32_t centered_y = static_cast<int32_t>(index % 129U) - 64;
    const int32_t centered_z = static_cast<int32_t>(index % 513U) - 256;
    const int32_t centered_vx = static_cast<int32_t>(index % 61U) - 30;
    const int32_t centered_vy = static_cast<int32_t>(index % 67U) - 33;
    const int32_t centered_vz = static_cast<int32_t>(index % 71U) - 35;
    const int32_t centered_aux0 = static_cast<int32_t>(index % 79U) - 39;
    const int32_t centered_aux1 = static_cast<int32_t>(index % 83U) - 41;
    const int32_t centered_aux2 = static_cast<int32_t>(index % 89U) - 44;
    const int32_t centered_aux3 = static_cast<int32_t>(index % 97U) - 48;
    const int32_t centered_cold0 = static_cast<int32_t>(index % 101U) - 50;
    const int32_t centered_cold1 = static_cast<int32_t>(index % 107U) - 53;
    const int32_t centered_cold2 = static_cast<int32_t>(index % 109U) - 54;
    const int32_t centered_cold3 = static_cast<int32_t>(index % 113U) - 56;

    ParticleAosStd430 particle{};
    particle.px = static_cast<float>(centered_x) * 0.125F;
    particle.py = static_cast<float>(centered_y) * 0.25F;
    particle.pz = static_cast<float>(centered_z) * 0.0625F;
    particle.mass = 1.0F + (static_cast<float>(index % 17U) * 0.01F);
    particle.vx = static_cast<float>(centered_vx) * 0.015F;
    particle.vy = static_cast<float>(centered_vy) * 0.02F;
    particle.vz = static_cast<float>(centered_vz) * 0.01F;
    particle.dt = 0.010F + (static_cast<float>(index % 11U) * 0.001F);
    particle.aux0 = static_cast<float>(centered_aux0) * 0.03125F;
    particle.aux1 = static_cast<float>(centered_aux1) * 0.015625F;
    particle.aux2 = static_cast<float>(centered_aux2) * 0.0078125F;
    particle.aux3 = static_cast<float>(centered_aux3) * 0.00390625F;
    particle.cold0 = static_cast<float>(centered_cold0) * 0.05F;
    particle.cold1 = static_cast<float>(centered_cold1) * 0.04F;
    particle.cold2 = static_cast<float>(centered_cold2) * 0.03F;
    particle.cold3 = static_cast<float>(centered_cold3) * 0.02F;
    return particle;
}

ParticleAosStd430 make_expected_particle(uint32_t index) {
    ParticleAosStd430 expected = make_seed_particle(index);
    const float adv_x = expected.vx * expected.dt;
    const float adv_y = expected.vy * expected.dt;
    const float adv_z = expected.vz * expected.dt;

    expected.px += adv_x;
    expected.py += adv_y;
    expected.pz += adv_z;
    expected.aux0 += (adv_x * 0.5F) + (adv_y * 0.25F);
    expected.aux1 += adv_z * 0.75F;
    expected.aux2 = (expected.aux2 * 0.999F) + (expected.aux0 * 0.001F);
    expected.aux3 += expected.aux2 * 0.01F;
    expected.mass += kMassIncrement + (expected.aux1 * 0.0000001F);
    return expected;
}

void fill_aos_seed_particles(ParticleAosStd430* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        particles[index] = make_seed_particle(index);
    }
}

bool validate_aos_particles(const ParticleAosStd430* particles, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 expected = make_expected_particle(index);
        const ParticleAosStd430 actual = particles[index];
        if (!nearly_equal(actual.px, expected.px) || !nearly_equal(actual.py, expected.py) ||
            !nearly_equal(actual.pz, expected.pz) || !nearly_equal(actual.mass, expected.mass) ||
            !nearly_equal(actual.vx, expected.vx) || !nearly_equal(actual.vy, expected.vy) ||
            !nearly_equal(actual.vz, expected.vz) || !nearly_equal(actual.dt, expected.dt) ||
            !nearly_equal(actual.aux0, expected.aux0) || !nearly_equal(actual.aux1, expected.aux1) ||
            !nearly_equal(actual.aux2, expected.aux2) || !nearly_equal(actual.aux3, expected.aux3) ||
            !nearly_equal(actual.cold0, expected.cold0) || !nearly_equal(actual.cold1, expected.cold1) ||
            !nearly_equal(actual.cold2, expected.cold2) || !nearly_equal(actual.cold3, expected.cold3)) {
            return false;
        }
    }

    return true;
}

std::array<float*, kFloatArraysForSoa> mapped_soa_arrays(const SoaPipelineResources& resources) {
    std::array<float*, kFloatArraysForSoa> arrays{};
    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        arrays[binding] = static_cast<float*>(resources.mapped_ptrs[binding]);
    }
    return arrays;
}

void fill_soa_seed_arrays(const std::array<float*, kFloatArraysForSoa>& arrays, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 particle = make_seed_particle(index);
        arrays[0U][index] = particle.px;
        arrays[1U][index] = particle.py;
        arrays[2U][index] = particle.pz;
        arrays[3U][index] = particle.mass;
        arrays[4U][index] = particle.vx;
        arrays[5U][index] = particle.vy;
        arrays[6U][index] = particle.vz;
        arrays[7U][index] = particle.dt;
        arrays[8U][index] = particle.aux0;
        arrays[9U][index] = particle.aux1;
        arrays[10U][index] = particle.aux2;
        arrays[11U][index] = particle.aux3;
        arrays[12U][index] = particle.cold0;
        arrays[13U][index] = particle.cold1;
        arrays[14U][index] = particle.cold2;
        arrays[15U][index] = particle.cold3;
    }
}

bool validate_soa_arrays(const std::array<float*, kFloatArraysForSoa>& arrays, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 expected = make_expected_particle(index);
        if (!nearly_equal(arrays[0U][index], expected.px) || !nearly_equal(arrays[1U][index], expected.py) ||
            !nearly_equal(arrays[2U][index], expected.pz) || !nearly_equal(arrays[3U][index], expected.mass) ||
            !nearly_equal(arrays[4U][index], expected.vx) || !nearly_equal(arrays[5U][index], expected.vy) ||
            !nearly_equal(arrays[6U][index], expected.vz) || !nearly_equal(arrays[7U][index], expected.dt) ||
            !nearly_equal(arrays[8U][index], expected.aux0) || !nearly_equal(arrays[9U][index], expected.aux1) ||
            !nearly_equal(arrays[10U][index], expected.aux2) || !nearly_equal(arrays[11U][index], expected.aux3) ||
            !nearly_equal(arrays[12U][index], expected.cold0) || !nearly_equal(arrays[13U][index], expected.cold1) ||
            !nearly_equal(arrays[14U][index], expected.cold2) || !nearly_equal(arrays[15U][index], expected.cold3)) {
            return false;
        }
    }

    return true;
}

void fill_aosoa_seed_data(float* data, uint32_t count, uint32_t block_size) {
    if (data == nullptr || block_size == 0U) {
        return;
    }

    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 particle = make_seed_particle(index);
        data[compute_aosoa_index(0U, index, block_size)] = particle.px;
        data[compute_aosoa_index(1U, index, block_size)] = particle.py;
        data[compute_aosoa_index(2U, index, block_size)] = particle.pz;
        data[compute_aosoa_index(3U, index, block_size)] = particle.mass;
        data[compute_aosoa_index(4U, index, block_size)] = particle.vx;
        data[compute_aosoa_index(5U, index, block_size)] = particle.vy;
        data[compute_aosoa_index(6U, index, block_size)] = particle.vz;
        data[compute_aosoa_index(7U, index, block_size)] = particle.dt;
        data[compute_aosoa_index(8U, index, block_size)] = particle.aux0;
        data[compute_aosoa_index(9U, index, block_size)] = particle.aux1;
        data[compute_aosoa_index(10U, index, block_size)] = particle.aux2;
        data[compute_aosoa_index(11U, index, block_size)] = particle.aux3;
        data[compute_aosoa_index(12U, index, block_size)] = particle.cold0;
        data[compute_aosoa_index(13U, index, block_size)] = particle.cold1;
        data[compute_aosoa_index(14U, index, block_size)] = particle.cold2;
        data[compute_aosoa_index(15U, index, block_size)] = particle.cold3;
    }
}

bool validate_aosoa_data(const float* data, uint32_t count, uint32_t block_size) {
    if (data == nullptr || block_size == 0U) {
        return false;
    }

    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 expected = make_expected_particle(index);
        if (!nearly_equal(data[compute_aosoa_index(0U, index, block_size)], expected.px) ||
            !nearly_equal(data[compute_aosoa_index(1U, index, block_size)], expected.py) ||
            !nearly_equal(data[compute_aosoa_index(2U, index, block_size)], expected.pz) ||
            !nearly_equal(data[compute_aosoa_index(3U, index, block_size)], expected.mass) ||
            !nearly_equal(data[compute_aosoa_index(4U, index, block_size)], expected.vx) ||
            !nearly_equal(data[compute_aosoa_index(5U, index, block_size)], expected.vy) ||
            !nearly_equal(data[compute_aosoa_index(6U, index, block_size)], expected.vz) ||
            !nearly_equal(data[compute_aosoa_index(7U, index, block_size)], expected.dt) ||
            !nearly_equal(data[compute_aosoa_index(8U, index, block_size)], expected.aux0) ||
            !nearly_equal(data[compute_aosoa_index(9U, index, block_size)], expected.aux1) ||
            !nearly_equal(data[compute_aosoa_index(10U, index, block_size)], expected.aux2) ||
            !nearly_equal(data[compute_aosoa_index(11U, index, block_size)], expected.aux3) ||
            !nearly_equal(data[compute_aosoa_index(12U, index, block_size)], expected.cold0) ||
            !nearly_equal(data[compute_aosoa_index(13U, index, block_size)], expected.cold1) ||
            !nearly_equal(data[compute_aosoa_index(14U, index, block_size)], expected.cold2) ||
            !nearly_equal(data[compute_aosoa_index(15U, index, block_size)], expected.cold3)) {
            return false;
        }
    }

    return true;
}

bool map_buffer_memory(VulkanContext& context, const BufferResource& buffer, const char* label, void*& mapped_ptr) {
    mapped_ptr = nullptr;
    const VkResult map_result = vkMapMemory(context.device(), buffer.memory, 0U, buffer.size, 0U, &mapped_ptr);
    if (map_result != VK_SUCCESS || mapped_ptr == nullptr) {
        std::cerr << "vkMapMemory failed for " << label << " with VkResult=" << map_result << ".\n";
        mapped_ptr = nullptr;
        return false;
    }

    return true;
}

bool create_aos_pipeline_resources(VulkanContext& context, const std::string& shader_path, VkDeviceSize buffer_size,
                                   AosPipelineResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.storage_buffer)) {
        std::cerr << "Failed to create AoS storage buffer.\n";
        return false;
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load AoS shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create AoS descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create AoS descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate AoS descriptor set.\n";
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
        std::cerr << "Failed to create AoS pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create AoS compute pipeline.\n";
        return false;
    }

    if (!map_buffer_memory(context, out_resources.storage_buffer, "AoS storage buffer", out_resources.mapped_ptr)) {
        return false;
    }

    return true;
}

bool create_soa_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                   VkDeviceSize buffer_size_per_array, SoaPipelineResources& out_resources) {
    out_resources.storage_buffers.assign(kFloatArraysForSoa, BufferResource{});
    out_resources.mapped_ptrs.fill(nullptr);

    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        if (!create_buffer_resource(context.physical_device(), context.device(), buffer_size_per_array,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    out_resources.storage_buffers[binding])) {
            std::cerr << "Failed to create SoA storage buffer for binding " << binding << ".\n";
            return false;
        }
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load SoA shader module: " << shader_path << "\n";
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(kFloatArraysForSoa);
    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        bindings.push_back(VkDescriptorSetLayoutBinding{binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
                                                        VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }

    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create SoA descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFloatArraysForSoa},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create SoA descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate SoA descriptor set.\n";
        return false;
    }

    std::vector<VulkanComputeUtils::DescriptorBufferBindingUpdate> updates;
    updates.reserve(kFloatArraysForSoa);
    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        const VkDescriptorBufferInfo info{
            out_resources.storage_buffers[binding].buffer,
            0U,
            out_resources.storage_buffers[binding].size,
        };
        updates.push_back(VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = binding,
            .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buffer_info = info,
        });
    }
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), out_resources.descriptor_set, updates);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{
            VK_SHADER_STAGE_COMPUTE_BIT,
            0U,
            sizeof(uint32_t),
        },
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create SoA pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create SoA compute pipeline.\n";
        return false;
    }

    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        if (!map_buffer_memory(context, out_resources.storage_buffers[binding], "SoA storage buffer",
                               out_resources.mapped_ptrs[binding])) {
            return false;
        }
    }

    return true;
}

bool create_aosoa_pipeline_resources(VulkanContext& context, const std::string& shader_path, VkDeviceSize buffer_size,
                                     AosoaPipelineResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.storage_buffer)) {
        std::cerr << "Failed to create AoSoA storage buffer.\n";
        return false;
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load AoSoA shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create AoSoA descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create AoSoA descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate AoSoA descriptor set.\n";
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
            sizeof(AosoaPushConstants),
        },
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create AoSoA pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create AoSoA compute pipeline.\n";
        return false;
    }

    if (!map_buffer_memory(context, out_resources.storage_buffer, "AoSoA storage buffer", out_resources.mapped_ptr)) {
        return false;
    }

    return true;
}

void cleanup_aos_resources(VulkanContext& context, AosPipelineResources& resources) {
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

void cleanup_soa_resources(VulkanContext& context, SoaPipelineResources& resources) {
    for (uint32_t binding = 0U; binding < kFloatArraysForSoa; ++binding) {
        if (resources.mapped_ptrs[binding] != nullptr && binding < resources.storage_buffers.size() &&
            resources.storage_buffers[binding].memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), resources.storage_buffers[binding].memory);
            resources.mapped_ptrs[binding] = nullptr;
        }
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
    for (auto& buffer : resources.storage_buffers) {
        destroy_buffer_resource(context.device(), buffer);
    }
    resources.storage_buffers.clear();
}

void cleanup_aosoa_resources(VulkanContext& context, AosoaPipelineResources& resources) {
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

double run_aos_or_soa_dispatch(VulkanContext& context, uint32_t particles, VkPipeline pipeline,
                               VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);
    if (group_count == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0U, 1U,
                                &descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(particles),
                           &particles);
        for (uint32_t dispatch_index = 0U; dispatch_index < kDispatchCount; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count, 1U, 1U);
        }
    });
}

double run_aosoa_dispatch(VulkanContext& context, uint32_t particles, uint32_t block_size, VkPipeline pipeline,
                          VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);
    const uint32_t block_count = compute_block_count(particles, block_size);
    if (group_count == 0U || block_count == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const AosoaPushConstants push_constants{
        .count = particles,
        .block_size = block_size,
        .block_count = block_count,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0U, 1U,
                                &descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(push_constants),
                           &push_constants);
        for (uint32_t dispatch_index = 0U; dispatch_index < kDispatchCount; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count, 1U, 1U);
        }
    });
}

double compute_throughput_elements_per_second(uint32_t particles, uint32_t dispatch_count, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double elements = static_cast<double>(particles) * static_cast<double>(dispatch_count);
    return (elements * 1000.0) / dispatch_gpu_ms;
}

double bytes_per_particle(LayoutVariant variant) {
    if (variant == LayoutVariant::kAos) {
        return static_cast<double>(kFloatsPerParticle * sizeof(float) * 2U);
    }

    constexpr uint32_t kReadFloats = 12U;
    constexpr uint32_t kWriteFloats = 8U;
    return static_cast<double>((kReadFloats + kWriteFloats) * sizeof(float));
}

double compute_effective_gbps(LayoutVariant variant, uint32_t particles, uint32_t dispatch_count,
                              double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes =
        bytes_per_particle(variant) * static_cast<double>(particles) * static_cast<double>(dispatch_count);
    return bytes / (dispatch_gpu_ms * 1.0e6);
}

std::string build_case_name(const std::string& variant, uint32_t particles) {
    return std::string(kExperimentId) + "_" + variant + "_particles_" + std::to_string(particles);
}

template <typename PrepareFn, typename DispatchFn, typename ValidateFn>
void run_variant_case(const BenchmarkRunner& runner, LayoutVariant layout_variant, const std::string& variant_name,
                      uint32_t particles, uint32_t group_count, PrepareFn prepare_fn, DispatchFn dispatch_fn,
                      ValidateFn validate_fn, const std::string& variant_notes, bool verbose_progress,
                      AosoaBlockedLayoutExperimentOutput& output) {
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant start: name=" << variant_name << ", particles=" << particles
                  << ", group_count_x=" << group_count << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    if (verbose_progress && runner.warmup_iterations() == 0) {
        std::cout << "[" << kExperimentId << "] warmup skipped for variant=" << variant_name
                  << ", particles=" << particles << "\n";
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        prepare_fn(particles);
        const double dispatch_ms = dispatch_fn(particles);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_fn(particles);
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

        prepare_fn(particles);
        const double dispatch_ms = dispatch_fn(particles);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_fn(particles);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        std::string notes;
        append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
        append_note(notes, "group_count_x=" + std::to_string(group_count));
        append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
        append_note(notes,
                    "bytes_per_particle=" + std::to_string(static_cast<uint64_t>(bytes_per_particle(layout_variant))));
        if (!variant_notes.empty()) {
            append_note(notes, variant_notes);
        }
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
            .gbps = compute_effective_gbps(layout_variant, particles, kDispatchCount, dispatch_ms),
            .correctness_pass = correctness,
            .notes = notes,
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(build_case_name(variant_name, particles), dispatch_samples);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant complete: name=" << variant_name << ", particles=" << particles
                  << ", median_ms=" << summary.median_ms << ", p95_ms=" << summary.p95_ms
                  << ", samples=" << summary.sample_count << "\n";
    }
    output.summary_results.push_back(summary);
}

} // namespace

AosoaBlockedLayoutExperimentOutput
run_aosoa_blocked_layout_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const AosoaBlockedLayoutExperimentConfig& config) {
    AosoaBlockedLayoutExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "AoSoA blocked layout experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string aos_shader = VulkanComputeUtils::resolve_shader_path(config.aos_shader_path, "07_aos.comp.spv");
    const std::string soa_shader = VulkanComputeUtils::resolve_shader_path(config.soa_shader_path, "07_soa.comp.spv");
    const std::string aosoa_shader =
        VulkanComputeUtils::resolve_shader_path(config.aosoa_shader_path, "07_aosoa_blocked.comp.spv");
    if (aos_shader.empty() || soa_shader.empty() || aosoa_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for AoSoA blocked layout experiment.\n";
        return output;
    }

    const uint64_t max_particles_u64 = config.max_buffer_bytes / (kFloatsPerParticle * sizeof(float));
    if (max_particles_u64 == 0U) {
        std::cerr << "Scratch buffer too small for AoSoA blocked layout experiment.\n";
        return output;
    }

    uint32_t max_particles =
        static_cast<uint32_t>(std::min<uint64_t>(max_particles_u64, std::numeric_limits<uint32_t>::max()));
    while (max_particles > 0U && compute_aosoa_storage_bytes_for_particles(max_particles, kAosoaBlockSizes.back()) >
                                     static_cast<VkDeviceSize>(config.max_buffer_bytes)) {
        --max_particles;
    }
    if (max_particles == 0U) {
        std::cerr << "Scratch buffer is too small after AoSoA block padding constraints.\n";
        return output;
    }

    const std::vector<uint32_t> particle_counts = make_particle_counts(max_particles);
    if (particle_counts.empty()) {
        std::cerr << "No legal particle counts available for AoSoA blocked layout experiment.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader AoS: " << aos_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader SoA: " << soa_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader AoSoA: " << aosoa_shader << "\n";
        std::cout << "[" << kExperimentId << "] Starting run with particles=" << particle_counts.size()
                  << ", variants=" << (2U + static_cast<uint32_t>(kAosoaBlockSizes.size()))
                  << ", local_size_x=" << kWorkgroupSize << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    AosPipelineResources aos_resources{};
    const VkDeviceSize aos_buffer_size =
        static_cast<VkDeviceSize>(particle_counts.back()) * kFloatsPerParticle * sizeof(float);
    if (!create_aos_pipeline_resources(context, aos_shader, aos_buffer_size, aos_resources)) {
        cleanup_aos_resources(context, aos_resources);
        return output;
    }

    SoaPipelineResources soa_resources{};
    const VkDeviceSize soa_array_size = static_cast<VkDeviceSize>(particle_counts.back()) * sizeof(float);
    if (!create_soa_pipeline_resources(context, soa_shader, soa_array_size, soa_resources)) {
        cleanup_aos_resources(context, aos_resources);
        cleanup_soa_resources(context, soa_resources);
        return output;
    }

    AosoaPipelineResources aosoa_resources{};
    const VkDeviceSize aosoa_buffer_size =
        compute_aosoa_storage_bytes_for_particles(particle_counts.back(), kAosoaBlockSizes.back());
    if (!create_aosoa_pipeline_resources(context, aosoa_shader, aosoa_buffer_size, aosoa_resources)) {
        cleanup_aos_resources(context, aos_resources);
        cleanup_soa_resources(context, soa_resources);
        cleanup_aosoa_resources(context, aosoa_resources);
        return output;
    }

    auto* aos_particles = static_cast<ParticleAosStd430*>(aos_resources.mapped_ptr);
    const std::array<float*, kFloatArraysForSoa> soa_arrays = mapped_soa_arrays(soa_resources);
    auto* aosoa_data = static_cast<float*>(aosoa_resources.mapped_ptr);
    if (aos_particles == nullptr || aosoa_data == nullptr || std::ranges::any_of(soa_arrays, [](const float* ptr) {
            return ptr == nullptr;
        })) {
        std::cerr << "Mapped pointers are missing for AoSoA blocked layout experiment resources.\n";
        cleanup_aos_resources(context, aos_resources);
        cleanup_soa_resources(context, soa_resources);
        cleanup_aosoa_resources(context, aosoa_resources);
        return output;
    }

    for (uint32_t particles : particle_counts) {
        const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);

        run_variant_case(
            runner, LayoutVariant::kAos, "aos", particles, group_count,
            [&](uint32_t count) {
                fill_aos_seed_particles(aos_particles, count);
            },
            [&](uint32_t count) {
                return run_aos_or_soa_dispatch(context, count, aos_resources.pipeline, aos_resources.pipeline_layout,
                                               aos_resources.descriptor_set);
            },
            [&](uint32_t count) {
                return validate_aos_particles(aos_particles, count);
            },
            "", verbose_progress, output);

        run_variant_case(
            runner, LayoutVariant::kSoa, "soa", particles, group_count,
            [&](uint32_t count) {
                fill_soa_seed_arrays(soa_arrays, count);
            },
            [&](uint32_t count) {
                return run_aos_or_soa_dispatch(context, count, soa_resources.pipeline, soa_resources.pipeline_layout,
                                               soa_resources.descriptor_set);
            },
            [&](uint32_t count) {
                return validate_soa_arrays(soa_arrays, count);
            },
            "", verbose_progress, output);

        for (uint32_t block_size : kAosoaBlockSizes) {
            const std::string variant_name = "aosoa_b" + std::to_string(block_size);
            const std::string variant_notes = "block_size=" + std::to_string(block_size);

            run_variant_case(
                runner, LayoutVariant::kAosoa, variant_name, particles, group_count,
                [&](uint32_t count) {
                    fill_aosoa_seed_data(aosoa_data, count, block_size);
                },
                [&](uint32_t count) {
                    return run_aosoa_dispatch(context, count, block_size, aosoa_resources.pipeline,
                                              aosoa_resources.pipeline_layout, aosoa_resources.descriptor_set);
                },
                [&](uint32_t count) {
                    return validate_aosoa_data(aosoa_data, count, block_size);
                },
                variant_notes, verbose_progress, output);
        }
    }

    cleanup_aos_resources(context, aos_resources);
    cleanup_soa_resources(context, soa_resources);
    cleanup_aosoa_resources(context, aosoa_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
