#include "experiments/aos_soa_experiment.hpp"

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

constexpr const char* kExperimentId = "06_aos_vs_soa";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kFloatArraysForSoa = 8U;
constexpr uint32_t kFloatsPerParticle = 8U;
constexpr uint32_t kDispatchCount = 1U;
constexpr float kMassIncrement = 0.000001F;
constexpr float kValidationEpsilon = 1.0e-5F;

enum class LayoutVariant : uint8_t {
    kAos = 0U,
    kSoa = 1U,
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
};

GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(ParticleAosStd430);
GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(ParticleAosStd430);
GPU_LAYOUT_ASSERT_SIZE(ParticleAosStd430, 32);
GPU_LAYOUT_ASSERT_ALIGNMENT(ParticleAosStd430, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, px, 0);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, py, 4);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, pz, 8);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, mass, 12);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vx, 16);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vy, 20);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, vz, 24);
GPU_LAYOUT_ASSERT_OFFSET(ParticleAosStd430, dt, 28);

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

std::vector<uint32_t> make_particle_counts(uint32_t max_particles) {
    const std::array<uint32_t, 3> base_counts = {1000000U, 5000000U, 10000000U};
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

    ParticleAosStd430 particle{};
    particle.px = static_cast<float>(centered_x) * 0.125F;
    particle.py = static_cast<float>(centered_y) * 0.25F;
    particle.pz = static_cast<float>(centered_z) * 0.0625F;
    particle.mass = 1.0F + (static_cast<float>(index % 17U) * 0.01F);
    particle.vx = static_cast<float>(centered_vx) * 0.015F;
    particle.vy = static_cast<float>(centered_vy) * 0.02F;
    particle.vz = static_cast<float>(centered_vz) * 0.01F;
    particle.dt = 0.010F + (static_cast<float>(index % 11U) * 0.001F);
    return particle;
}

ParticleAosStd430 make_expected_particle(uint32_t index) {
    ParticleAosStd430 expected = make_seed_particle(index);
    expected.px += expected.vx * expected.dt;
    expected.py += expected.vy * expected.dt;
    expected.pz += expected.vz * expected.dt;
    expected.mass += kMassIncrement;
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
            !nearly_equal(actual.vz, expected.vz) || !nearly_equal(actual.dt, expected.dt)) {
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
    }
}

bool validate_soa_arrays(const std::array<float*, kFloatArraysForSoa>& arrays, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        const ParticleAosStd430 expected = make_expected_particle(index);
        if (!nearly_equal(arrays[0U][index], expected.px) || !nearly_equal(arrays[1U][index], expected.py) ||
            !nearly_equal(arrays[2U][index], expected.pz) || !nearly_equal(arrays[3U][index], expected.mass) ||
            !nearly_equal(arrays[4U][index], expected.vx) || !nearly_equal(arrays[5U][index], expected.vy) ||
            !nearly_equal(arrays[6U][index], expected.vz) || !nearly_equal(arrays[7U][index], expected.dt)) {
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

double run_dispatch(VulkanContext& context, uint32_t particles, VkPipeline pipeline, VkPipelineLayout pipeline_layout,
                    VkDescriptorSet descriptor_set) {
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

    constexpr uint32_t kSoaReadFloats = kFloatArraysForSoa;
    constexpr uint32_t kSoaWriteFloats = 4U;
    return static_cast<double>((kSoaReadFloats + kSoaWriteFloats) * sizeof(float));
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

template <typename PrepareFn, typename ValidateFn>
void run_variant_case(VulkanContext& context, const BenchmarkRunner& runner, LayoutVariant layout_variant,
                      const std::string& variant_name, uint32_t particles, VkPipeline pipeline,
                      VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set, PrepareFn prepare_inputs,
                      ValidateFn validate_outputs, AosSoaExperimentOutput& output) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(particles, kWorkgroupSize);
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        prepare_inputs(particles);
        const double dispatch_ms = run_dispatch(context, particles, pipeline, pipeline_layout, descriptor_set);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(particles);
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] warmup issue for variant=" << variant_name
                      << ", particles=" << particles << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        prepare_inputs(particles);
        const double dispatch_ms = run_dispatch(context, particles, pipeline, pipeline_layout, descriptor_set);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(particles);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        std::string notes;
        append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
        append_note(notes, "group_count_x=" + std::to_string(group_count));
        append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
        append_note(notes,
                    "bytes_per_particle=" + std::to_string(static_cast<uint64_t>(bytes_per_particle(layout_variant))));
        if (!dispatch_ok) {
            append_note(notes, "dispatch_ms_non_finite");
        }
        if (!data_ok) {
            append_note(notes, "correctness_mismatch");
        }

        const bool correctness = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness;
        dispatch_samples.push_back(dispatch_ms);

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

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(build_case_name(variant_name, particles), dispatch_samples));
}

} // namespace

AosSoaExperimentOutput run_aos_soa_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                              const AosSoaExperimentConfig& config) {
    AosSoaExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "AoS vs SoA experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string aos_shader = VulkanComputeUtils::resolve_shader_path(config.aos_shader_path, "06_aos.comp.spv");
    const std::string soa_shader = VulkanComputeUtils::resolve_shader_path(config.soa_shader_path, "06_soa.comp.spv");
    if (aos_shader.empty() || soa_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for AoS vs SoA experiment.\n";
        return output;
    }

    const uint64_t max_particles_u64 = config.max_buffer_bytes / (kFloatsPerParticle * sizeof(float));
    if (max_particles_u64 == 0U) {
        std::cerr << "Scratch buffer too small for AoS vs SoA experiment.\n";
        return output;
    }

    const uint32_t max_particles =
        static_cast<uint32_t>(std::min<uint64_t>(max_particles_u64, std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> particle_counts = make_particle_counts(max_particles);
    if (particle_counts.empty()) {
        std::cerr << "No legal particle counts available for AoS vs SoA experiment.\n";
        return output;
    }

    std::cout << "[" << kExperimentId << "] Shader AoS: " << aos_shader << "\n";
    std::cout << "[" << kExperimentId << "] Shader SoA: " << soa_shader << "\n";
    std::cout << "[" << kExperimentId << "] Starting run with particles=" << particle_counts.size()
              << ", variants=2, local_size_x=" << kWorkgroupSize << ", warmup_iterations=" << runner.warmup_iterations()
              << ", timed_iterations=" << runner.timed_iterations() << "\n";

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

    auto* aos_particles = static_cast<ParticleAosStd430*>(aos_resources.mapped_ptr);
    const std::array<float*, kFloatArraysForSoa> soa_arrays = mapped_soa_arrays(soa_resources);
    if (aos_particles == nullptr || std::ranges::any_of(soa_arrays, [](const float* ptr) {
            return ptr == nullptr;
        })) {
        std::cerr << "Mapped pointers are missing for AoS vs SoA experiment resources.\n";
        cleanup_aos_resources(context, aos_resources);
        cleanup_soa_resources(context, soa_resources);
        return output;
    }

    for (uint32_t particles : particle_counts) {
        run_variant_case(
            context, runner, LayoutVariant::kAos, "aos", particles, aos_resources.pipeline,
            aos_resources.pipeline_layout, aos_resources.descriptor_set,
            [&](uint32_t count) {
                fill_aos_seed_particles(aos_particles, count);
            },
            [&](uint32_t count) {
                return validate_aos_particles(aos_particles, count);
            },
            output);

        run_variant_case(
            context, runner, LayoutVariant::kSoa, "soa", particles, soa_resources.pipeline,
            soa_resources.pipeline_layout, soa_resources.descriptor_set,
            [&](uint32_t count) {
                fill_soa_seed_arrays(soa_arrays, count);
            },
            [&](uint32_t count) {
                return validate_soa_arrays(soa_arrays, count);
            },
            output);
    }

    cleanup_aos_resources(context, aos_resources);
    cleanup_soa_resources(context, soa_resources);

    std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
              << ", rows=" << output.rows.size()
              << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    return output;
}
