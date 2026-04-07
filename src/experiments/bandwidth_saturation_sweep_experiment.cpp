#include "experiments/bandwidth_saturation_sweep_experiment.hpp"

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
#include <vector>

namespace {

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "15_bandwidth_saturation_sweep";
constexpr uint32_t kLocalSizeX = 256U;
constexpr VkDeviceSize kBytesPerMiB = 1024ULL * 1024ULL;
constexpr std::array<uint32_t, 17> kSweepSizesMiB = {
    1U, 2U, 4U, 8U, 16U, 24U, 32U, 48U, 64U, 96U, 128U, 192U, 256U, 384U, 512U, 768U, 1024U,
};
constexpr float kWriteOnlySentinel = -12345.0F;
constexpr float kCopyDestinationSentinel = -23456.0F;
constexpr float kReadOnlyProbeSentinel = -34567.0F;
constexpr uint32_t kReadOnlyProbeCount = 4U;

enum class MemoryMode : std::uint8_t {
    kReadOnly = 0U,
    kWriteOnly = 1U,
    kReadWriteCopy = 2U,
};

struct ModeDescriptor {
    MemoryMode mode = MemoryMode::kReadOnly;
    const char* variant_name = "";
    uint32_t bytes_per_element = 0U;
};

constexpr std::array<ModeDescriptor, 3> kModes = {
    ModeDescriptor{
        .mode = MemoryMode::kReadOnly,
        .variant_name = "read_only",
        .bytes_per_element = static_cast<uint32_t>(sizeof(float)),
    },
    ModeDescriptor{
        .mode = MemoryMode::kWriteOnly,
        .variant_name = "write_only",
        .bytes_per_element = static_cast<uint32_t>(sizeof(float)),
    },
    ModeDescriptor{
        .mode = MemoryMode::kReadWriteCopy,
        .variant_name = "read_write_copy",
        .bytes_per_element = static_cast<uint32_t>(sizeof(float) * 2U),
    },
};

struct ExperimentBufferResources {
    BufferResource src_device{};
    BufferResource dst_device{};
    BufferResource staging{};
};

struct VariantPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

std::string build_case_name(const std::string& variant, uint32_t problem_size) {
    return std::string(kExperimentId) + "_" + variant + "_size_" + std::to_string(problem_size);
}

float source_pattern_value(uint32_t index) {
    return static_cast<float>(static_cast<int32_t>(index % 2048U) - 1024);
}

void fill_source_pattern(float* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_sentinel(float* values, uint32_t element_count, float sentinel_value) {
    std::fill_n(values, element_count, sentinel_value);
}

bool validate_source_pattern(const float* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != source_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

bool validate_write_pattern(const float* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != static_cast<float>(index)) {
            return false;
        }
    }
    return true;
}

uint32_t read_only_probe_source_index(uint32_t probe_slot, uint32_t element_count) {
    switch (probe_slot) {
    case 0U:
        return 0U;
    case 1U:
        return element_count / 3U;
    case 2U:
        return (element_count * 2U) / 3U;
    case 3U:
        return element_count - 1U;
    default:
        return 0U;
    }
}

void fill_read_only_probe_sentinel(float* values) {
    std::fill_n(values, kReadOnlyProbeCount, kReadOnlyProbeSentinel);
}

bool validate_read_only_probe(const float* values, uint32_t element_count) {
    if (element_count == 0U) {
        return false;
    }

    for (uint32_t probe_slot = 0U; probe_slot < kReadOnlyProbeCount; ++probe_slot) {
        const uint32_t source_index = read_only_probe_source_index(probe_slot, element_count);
        if (values[probe_slot] != source_pattern_value(source_index)) {
            return false;
        }
    }

    return true;
}

double compute_effective_gbps(uint32_t problem_size, uint32_t bytes_per_element, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes = static_cast<double>(problem_size) * static_cast<double>(bytes_per_element);
    return bytes / (dispatch_gpu_ms * 1.0e6);
}

void destroy_variant_pipeline_resources(VulkanContext& context, VariantPipelineResources& resources) {
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

void destroy_experiment_buffer_resources(VulkanContext& context, ExperimentBufferResources& resources) {
    destroy_buffer_resource(context.device(), resources.staging);
    destroy_buffer_resource(context.device(), resources.dst_device);
    destroy_buffer_resource(context.device(), resources.src_device);
}

bool create_experiment_buffer_resources(VulkanContext& context, VkDeviceSize max_buffer_size,
                                        ExperimentBufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), max_buffer_size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.src_device)) {
        std::cerr << "Failed to create source device buffer for bandwidth saturation sweep experiment.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), max_buffer_size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.dst_device)) {
        std::cerr << "Failed to create destination device buffer for bandwidth saturation sweep experiment.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), max_buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging)) {
        std::cerr << "Failed to create staging buffer for bandwidth saturation sweep experiment.\n";
        return false;
    }

    return true;
}

bool create_read_only_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                         const BufferResource& src_device, const BufferResource& probe_device,
                                         VariantPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load read-only shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create read-only descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create read-only descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate read-only descriptor set.\n";
        return false;
    }

    const VkDescriptorBufferInfo src_info{src_device.buffer, 0U, src_device.size};
    const VkDescriptorBufferInfo probe_info{probe_device.buffer, 0U, probe_device.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out_resources.descriptor_set,
        {
            VulkanComputeUtils::DescriptorBufferBindingUpdate{
                .binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = src_info},
            VulkanComputeUtils::DescriptorBufferBindingUpdate{
                .binding = 1U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = probe_info},
        });

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(uint32_t)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create read-only pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create read-only compute pipeline.\n";
        return false;
    }

    return true;
}

bool create_write_only_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                          const BufferResource& dst_device, VariantPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load write-only shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create write-only descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create write-only descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate write-only descriptor set.\n";
        return false;
    }

    const VkDescriptorBufferInfo dst_info{dst_device.buffer, 0U, dst_device.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out_resources.descriptor_set,
        {VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = dst_info}});

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(uint32_t)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create write-only pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create write-only compute pipeline.\n";
        return false;
    }

    return true;
}

bool create_copy_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                    const BufferResource& src_device, const BufferResource& dst_device,
                                    VariantPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load read+write copy shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create read+write copy descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create read+write copy descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate read+write copy descriptor set.\n";
        return false;
    }

    const VkDescriptorBufferInfo src_info{src_device.buffer, 0U, src_device.size};
    const VkDescriptorBufferInfo dst_info{dst_device.buffer, 0U, dst_device.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out_resources.descriptor_set,
        {
            VulkanComputeUtils::DescriptorBufferBindingUpdate{
                .binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = src_info},
            VulkanComputeUtils::DescriptorBufferBindingUpdate{
                .binding = 1U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = dst_info},
        });

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(uint32_t)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create read+write copy pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create read+write copy compute pipeline.\n";
        return false;
    }

    return true;
}

double run_upload_stage(VulkanContext& context, const BufferResource& staging_buffer, const BufferResource& dst_buffer,
                        VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, staging_buffer.buffer, dst_buffer.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, dst_buffer.buffer,
                                                                                bytes);
    });
}

double run_readback_stage(VulkanContext& context, const BufferResource& src_buffer,
                          const BufferResource& staging_buffer, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, src_buffer.buffer, staging_buffer.buffer, 1U, &copy_region);
    });
}

double run_read_only_dispatch_stage(VulkanContext& context, const VariantPipelineResources& resources,
                                    const BufferResource& probe_buffer, VkDeviceSize probe_bytes,
                                    uint32_t group_count_x, uint32_t element_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(element_count), &element_count);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, probe_buffer.buffer,
                                                                          probe_bytes);
    });
}

double run_write_only_dispatch_stage(VulkanContext& context, const VariantPipelineResources& resources,
                                     const BufferResource& dst_buffer, VkDeviceSize bytes, uint32_t group_count_x,
                                     uint32_t element_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(element_count), &element_count);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, dst_buffer.buffer, bytes);
    });
}

double run_copy_dispatch_stage(VulkanContext& context, const VariantPipelineResources& resources,
                               const BufferResource& dst_buffer, VkDeviceSize bytes, uint32_t group_count_x,
                               uint32_t element_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(element_count), &element_count);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, dst_buffer.buffer, bytes);
    });
}

std::vector<uint32_t> make_problem_sizes(std::size_t max_buffer_bytes, uint32_t max_elements,
                                         uint32_t max_group_count_x) {
    std::vector<uint32_t> sizes;
    sizes.reserve(kSweepSizesMiB.size());

    for (const uint32_t size_mib : kSweepSizesMiB) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(size_mib) * kBytesPerMiB;
        if (bytes > static_cast<VkDeviceSize>(max_buffer_bytes)) {
            continue;
        }

        const auto elements_u64 = static_cast<uint64_t>(bytes / sizeof(float));
        if (elements_u64 == 0U || elements_u64 > max_elements || elements_u64 > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        const auto element_count = static_cast<uint32_t>(elements_u64);
        const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kLocalSizeX);
        if (group_count_x == 0U || group_count_x > max_group_count_x) {
            continue;
        }

        sizes.push_back(element_count);
    }

    return sizes;
}

} // namespace

BandwidthSaturationSweepExperimentOutput
run_bandwidth_saturation_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                          const BandwidthSaturationSweepExperimentConfig& config) {
    BandwidthSaturationSweepExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "Bandwidth saturation sweep experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string read_only_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.read_only_shader_path, "15_bandwidth_read_only.comp.spv");
    if (read_only_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for bandwidth saturation sweep read-only mode.\n";
        return output;
    }

    const std::string write_only_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.write_only_shader_path, "15_bandwidth_write_only.comp.spv");
    if (write_only_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for bandwidth saturation sweep write-only mode.\n";
        return output;
    }

    const std::string read_write_copy_shader_path = VulkanComputeUtils::resolve_shader_path(
        config.read_write_copy_shader_path, "15_bandwidth_read_write_copy.comp.spv");
    if (read_write_copy_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for bandwidth saturation sweep read+write copy mode.\n";
        return output;
    }

    const auto buffer_limited_elements_u64 = static_cast<uint64_t>(config.max_buffer_bytes / sizeof(float));
    if (buffer_limited_elements_u64 == 0U) {
        std::cerr << "Scratch buffer too small for bandwidth saturation sweep experiment.\n";
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint64_t dispatch_limited_elements_u64 =
        static_cast<uint64_t>(device_properties.limits.maxComputeWorkGroupCount[0]) * kLocalSizeX;
    const uint64_t effective_max_elements_u64 = std::min({buffer_limited_elements_u64, dispatch_limited_elements_u64,
                                                          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});

    if (effective_max_elements_u64 == 0U) {
        std::cerr << "No legal problem sizes available for bandwidth saturation sweep experiment.\n";
        return output;
    }

    if (effective_max_elements_u64 < buffer_limited_elements_u64) {
        std::cerr << "Bandwidth saturation sweep clamped by maxComputeWorkGroupCount[0].\n";
    }

    const auto max_elements = static_cast<uint32_t>(effective_max_elements_u64);
    const std::vector<uint32_t> problem_sizes =
        make_problem_sizes(config.max_buffer_bytes, max_elements, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (problem_sizes.empty()) {
        std::cerr << "Scratch buffer is too small for the planned dense [1 MiB, 1 GiB] sweep.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Read-only shader: " << read_only_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] Write-only shader: " << write_only_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] Read+write copy shader: " << read_write_copy_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] Starting run with problem_sizes=" << problem_sizes.size()
                  << ", local_size_x=" << kLocalSizeX << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const VkDeviceSize max_buffer_size = static_cast<VkDeviceSize>(problem_sizes.back()) * sizeof(float);
    ExperimentBufferResources buffers{};
    if (!create_experiment_buffer_resources(context, max_buffer_size, buffers)) {
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    VariantPipelineResources read_only_resources{};
    if (!create_read_only_pipeline_resources(context, read_only_shader_path, buffers.src_device, buffers.dst_device,
                                             read_only_resources)) {
        destroy_variant_pipeline_resources(context, read_only_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    VariantPipelineResources write_only_resources{};
    if (!create_write_only_pipeline_resources(context, write_only_shader_path, buffers.dst_device,
                                              write_only_resources)) {
        destroy_variant_pipeline_resources(context, write_only_resources);
        destroy_variant_pipeline_resources(context, read_only_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    VariantPipelineResources copy_resources{};
    if (!create_copy_pipeline_resources(context, read_write_copy_shader_path, buffers.src_device, buffers.dst_device,
                                        copy_resources)) {
        destroy_variant_pipeline_resources(context, copy_resources);
        destroy_variant_pipeline_resources(context, write_only_resources);
        destroy_variant_pipeline_resources(context, read_only_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    void* mapped_staging = nullptr;
    const VkResult map_result =
        vkMapMemory(context.device(), buffers.staging.memory, 0U, buffers.staging.size, 0U, &mapped_staging);
    if (map_result != VK_SUCCESS || mapped_staging == nullptr) {
        std::cerr << "vkMapMemory failed for bandwidth saturation sweep staging buffer with error code " << map_result
                  << "\n";
        destroy_variant_pipeline_resources(context, copy_resources);
        destroy_variant_pipeline_resources(context, write_only_resources);
        destroy_variant_pipeline_resources(context, read_only_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    auto* staging_values = static_cast<float*>(mapped_staging);

    const std::size_t total_case_count = problem_sizes.size() * kModes.size();
    std::size_t completed_case_count = 0U;

    for (const uint32_t problem_size : problem_sizes) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(problem_size) * sizeof(float);
        const VkDeviceSize probe_bytes = static_cast<VkDeviceSize>(kReadOnlyProbeCount) * sizeof(float);
        const auto size_mib = static_cast<uint32_t>(bytes / kBytesPerMiB);
        const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, kLocalSizeX);

        for (const ModeDescriptor& mode_descriptor : kModes) {
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Case " << (completed_case_count + 1U) << "/" << total_case_count
                          << ": variant=" << mode_descriptor.variant_name << ", problem_size=" << problem_size
                          << ", size_mib=" << size_mib << ", group_count_x=" << group_count_x << "\n";
            }

            std::vector<double> dispatch_samples;
            dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

            for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                if (mode_descriptor.mode == MemoryMode::kReadOnly) {
                    fill_source_pattern(staging_values, problem_size);
                    const double upload_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_read_only_probe_sentinel(staging_values);
                    const double upload_probe_ms =
                        run_upload_stage(context, buffers.staging, buffers.dst_device, probe_bytes);
                    const double dispatch_ms = run_read_only_dispatch_stage(
                        context, read_only_resources, buffers.dst_device, probe_bytes, group_count_x, problem_size);
                    const double readback_ms =
                        run_readback_stage(context, buffers.dst_device, buffers.staging, probe_bytes);
                    const bool probe_ok = validate_read_only_probe(staging_values, problem_size);

                    if (!std::isfinite(upload_ms) || !std::isfinite(upload_probe_ms) || !std::isfinite(dispatch_ms) ||
                        !std::isfinite(readback_ms) || !probe_ok) {
                        std::cerr << "Warmup produced non-finite timing value in read-only mode.\n";
                    }

                    if (verbose_progress) {
                        const bool warmup_ok = std::isfinite(upload_ms) && std::isfinite(upload_probe_ms) &&
                                               std::isfinite(dispatch_ms) && std::isfinite(readback_ms) && probe_ok;
                        std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                                  << runner.warmup_iterations() << " variant=" << mode_descriptor.variant_name
                                  << ", problem_size=" << problem_size << ", size_mib=" << size_mib
                                  << ", upload_ms=" << upload_ms << ", upload_probe_ms=" << upload_probe_ms
                                  << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                                  << ", correctness=" << (warmup_ok ? "pass" : "fail") << "\n";
                    }
                } else if (mode_descriptor.mode == MemoryMode::kWriteOnly) {
                    fill_sentinel(staging_values, problem_size, kWriteOnlySentinel);
                    const double upload_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    const double dispatch_ms = run_write_only_dispatch_stage(
                        context, write_only_resources, buffers.dst_device, bytes, group_count_x, problem_size);
                    const double readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);

                    if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms)) {
                        std::cerr << "Warmup produced non-finite timing value in write-only mode.\n";
                    }

                    if (verbose_progress) {
                        const bool warmup_ok =
                            std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                        std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                                  << runner.warmup_iterations() << " variant=" << mode_descriptor.variant_name
                                  << ", problem_size=" << problem_size << ", size_mib=" << size_mib
                                  << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                                  << ", readback_ms=" << readback_ms
                                  << ", correctness=" << (warmup_ok ? "pass" : "fail") << "\n";
                    }
                } else {
                    fill_source_pattern(staging_values, problem_size);
                    const double upload_src_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_sentinel(staging_values, problem_size, kCopyDestinationSentinel);
                    const double upload_dst_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    const double dispatch_ms = run_copy_dispatch_stage(context, copy_resources, buffers.dst_device,
                                                                       bytes, group_count_x, problem_size);
                    const double readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);

                    if (!std::isfinite(upload_src_ms) || !std::isfinite(upload_dst_ms) || !std::isfinite(dispatch_ms) ||
                        !std::isfinite(readback_ms)) {
                        std::cerr << "Warmup produced non-finite timing value in read+write copy mode.\n";
                    }

                    if (verbose_progress) {
                        const bool warmup_ok = std::isfinite(upload_src_ms) && std::isfinite(upload_dst_ms) &&
                                               std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                        std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                                  << runner.warmup_iterations() << " variant=" << mode_descriptor.variant_name
                                  << ", problem_size=" << problem_size << ", size_mib=" << size_mib
                                  << ", upload_src_ms=" << upload_src_ms << ", upload_dst_ms=" << upload_dst_ms
                                  << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                                  << ", correctness=" << (warmup_ok ? "pass" : "fail") << "\n";
                    }
                }
            }

            for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                double upload_ms = std::numeric_limits<double>::quiet_NaN();
                double upload_dst_ms = std::numeric_limits<double>::quiet_NaN();
                double dispatch_ms = std::numeric_limits<double>::quiet_NaN();
                double readback_ms = std::numeric_limits<double>::quiet_NaN();

                const auto start = std::chrono::high_resolution_clock::now();

                if (mode_descriptor.mode == MemoryMode::kReadOnly) {
                    fill_source_pattern(staging_values, problem_size);
                    upload_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_read_only_probe_sentinel(staging_values);
                    upload_dst_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, probe_bytes);
                    dispatch_ms = run_read_only_dispatch_stage(context, read_only_resources, buffers.dst_device,
                                                               probe_bytes, group_count_x, problem_size);
                    readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, probe_bytes);
                } else if (mode_descriptor.mode == MemoryMode::kWriteOnly) {
                    fill_sentinel(staging_values, problem_size, kWriteOnlySentinel);
                    upload_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    dispatch_ms = run_write_only_dispatch_stage(context, write_only_resources, buffers.dst_device,
                                                                bytes, group_count_x, problem_size);
                    readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);
                } else {
                    fill_source_pattern(staging_values, problem_size);
                    upload_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_sentinel(staging_values, problem_size, kCopyDestinationSentinel);
                    upload_dst_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    dispatch_ms = run_copy_dispatch_stage(context, copy_resources, buffers.dst_device, bytes,
                                                          group_count_x, problem_size);
                    readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);
                }

                const auto end = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

                const bool upload_ok = std::isfinite(upload_ms);
                const bool upload_dst_ok =
                    std::isfinite(upload_dst_ms) || (mode_descriptor.mode == MemoryMode::kWriteOnly);
                const bool dispatch_ok = std::isfinite(dispatch_ms);
                const bool readback_ok = std::isfinite(readback_ms);

                bool data_ok = false;
                switch (mode_descriptor.mode) {
                case MemoryMode::kReadOnly:
                    data_ok = validate_read_only_probe(staging_values, problem_size);
                    break;
                case MemoryMode::kReadWriteCopy:
                    data_ok = validate_source_pattern(staging_values, problem_size);
                    break;
                case MemoryMode::kWriteOnly:
                    data_ok = validate_write_pattern(staging_values, problem_size);
                    break;
                }

                std::string notes;
                append_note(notes, "size_bytes=" + std::to_string(static_cast<unsigned long long>(bytes)));
                append_note(notes, "size_mib=" + std::to_string(size_mib));
                append_note(notes, "bytes_per_element=" + std::to_string(mode_descriptor.bytes_per_element));
                append_note(notes, "local_size_x=" + std::to_string(kLocalSizeX));
                append_note(notes, "group_count_x=" + std::to_string(group_count_x));
                if (mode_descriptor.mode == MemoryMode::kReadOnly) {
                    append_note(notes, "validation_probe_count=" + std::to_string(kReadOnlyProbeCount));
                }
                if (!upload_ok) {
                    append_note(notes, "upload_ms_non_finite");
                }
                if (!upload_dst_ok) {
                    append_note(notes, "upload_dst_ms_non_finite");
                }
                if (!dispatch_ok) {
                    append_note(notes, "dispatch_ms_non_finite");
                }
                if (!readback_ok) {
                    append_note(notes, "readback_ms_non_finite");
                }
                if (!data_ok) {
                    append_note(notes, "correctness_mismatch");
                }

                const bool correctness = upload_ok && upload_dst_ok && dispatch_ok && readback_ok && data_ok;
                dispatch_samples.push_back(dispatch_ms);
                if (verbose_progress) {
                    std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/"
                              << runner.timed_iterations() << " variant=" << mode_descriptor.variant_name
                              << ", problem_size=" << problem_size << ", size_mib=" << size_mib
                              << ", upload_ms=" << upload_ms << ", upload_dst_ms=" << upload_dst_ms
                              << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                              << ", end_to_end_ms=" << end_to_end_ms.count()
                              << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                }
                output.rows.push_back(BenchmarkMeasurementRow{
                    .experiment_id = kExperimentId,
                    .variant = mode_descriptor.variant_name,
                    .problem_size = problem_size,
                    .dispatch_count = 1U,
                    .iteration = iteration,
                    .gpu_ms = dispatch_ms,
                    .end_to_end_ms = end_to_end_ms.count(),
                    .throughput = compute_throughput_elements_per_second(problem_size, 1U, dispatch_ms),
                    .gbps = compute_effective_gbps(problem_size, mode_descriptor.bytes_per_element, dispatch_ms),
                    .correctness_pass = correctness,
                    .notes = notes,
                });
                output.all_points_correct = output.all_points_correct && correctness;
            }

            const BenchmarkResult summary = BenchmarkRunner::summarize_samples(
                build_case_name(mode_descriptor.variant_name, problem_size), dispatch_samples);
            output.summary_results.push_back(summary);
            ++completed_case_count;
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Completed case " << completed_case_count << "/"
                          << total_case_count << ": variant=" << mode_descriptor.variant_name
                          << ", problem_size=" << problem_size << ", size_mib=" << size_mib
                          << ", samples=" << summary.sample_count << ", median_gpu_ms=" << summary.median_ms << "\n";
            }
        }
    }

    vkUnmapMemory(context.device(), buffers.staging.memory);
    destroy_variant_pipeline_resources(context, copy_resources);
    destroy_variant_pipeline_resources(context, write_only_resources);
    destroy_variant_pipeline_resources(context, read_only_resources);
    destroy_experiment_buffer_resources(context, buffers);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
