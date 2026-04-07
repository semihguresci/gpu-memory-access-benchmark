#include "experiments/local_size_sweep_experiment.hpp"

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

using ExperimentMetrics::compute_effective_gbps;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "02_local_size_sweep";
constexpr uint32_t kMinProblemPower = 14;
constexpr uint32_t kMaxProblemPower = 24;
constexpr float kWriteSentinel = -1.0F;
constexpr float kNoopSentinel = -2.0F;
constexpr std::array<uint32_t, 6> kLocalSizeCandidates = {32U, 64U, 128U, 256U, 512U, 1024U};

struct SharedWriteResources {
    BufferResource device_storage{};
    BufferResource upload_staging{};
    BufferResource readback_staging{};
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
};

struct SharedNoopResources {
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
};

struct LocalSizePipelineResources {
    uint32_t local_size_x = 0U;
    VkShaderModule write_shader_module = VK_NULL_HANDLE;
    VkPipeline write_pipeline = VK_NULL_HANDLE;
    VkShaderModule noop_shader_module = VK_NULL_HANDLE;
    VkPipeline noop_pipeline = VK_NULL_HANDLE;
};

std::string write_shader_name(uint32_t local_size_x) {
    return "02_local_size_" + std::to_string(local_size_x) + "_write.comp.spv";
}

std::string noop_shader_name(uint32_t local_size_x) {
    return "02_local_size_" + std::to_string(local_size_x) + "_noop.comp.spv";
}

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

bool is_local_size_candidate_legal(const VkPhysicalDeviceProperties& device_properties, uint32_t local_size_x) {
    if (local_size_x == 0U) {
        return false;
    }

    if (local_size_x > device_properties.limits.maxComputeWorkGroupSize[0]) {
        return false;
    }

    if (local_size_x > device_properties.limits.maxComputeWorkGroupInvocations) {
        return false;
    }

    return true;
}

std::vector<uint32_t> make_base_problem_sizes(uint32_t max_elements) {
    std::vector<uint32_t> sizes;
    sizes.reserve(kMaxProblemPower - kMinProblemPower + 1U);

    for (uint32_t power = kMinProblemPower; power <= kMaxProblemPower; ++power) {
        const uint64_t value = 1ULL << power;
        if (value <= max_elements && value <= std::numeric_limits<uint32_t>::max()) {
            sizes.push_back(static_cast<uint32_t>(value));
        }
    }

    return sizes;
}

std::vector<uint32_t> filter_problem_sizes_for_local_size(const std::vector<uint32_t>& base_problem_sizes,
                                                          uint32_t local_size_x, uint32_t max_group_count_x) {
    std::vector<uint32_t> sizes;
    sizes.reserve(base_problem_sizes.size());

    for (const uint32_t problem_size : base_problem_sizes) {
        const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, local_size_x);
        if (group_count_x <= max_group_count_x) {
            sizes.push_back(problem_size);
        }
    }

    return sizes;
}

std::string build_case_name(const std::string& variant_name, uint32_t local_size_x, uint32_t problem_size,
                            uint32_t dispatch_count) {
    return std::string(kExperimentId) + "_" + variant_name + "_ls_" + std::to_string(local_size_x) + "_size_" +
           std::to_string(problem_size) + "_dispatches_" + std::to_string(dispatch_count);
}

void destroy_shared_write_resources(VulkanContext& context, SharedWriteResources& resources) {
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

    destroy_buffer_resource(context.device(), resources.readback_staging);
    destroy_buffer_resource(context.device(), resources.upload_staging);
    destroy_buffer_resource(context.device(), resources.device_storage);
    resources.descriptor_set = VK_NULL_HANDLE;
}

void destroy_shared_noop_resources(VulkanContext& context, SharedNoopResources& resources) {
    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }
}

void destroy_local_size_pipelines(VulkanContext& context, std::vector<LocalSizePipelineResources>& resources) {
    for (LocalSizePipelineResources& per_local_size : resources) {
        if (per_local_size.noop_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device(), per_local_size.noop_pipeline, nullptr);
            per_local_size.noop_pipeline = VK_NULL_HANDLE;
        }

        if (per_local_size.noop_shader_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device(), per_local_size.noop_shader_module, nullptr);
            per_local_size.noop_shader_module = VK_NULL_HANDLE;
        }

        if (per_local_size.write_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context.device(), per_local_size.write_pipeline, nullptr);
            per_local_size.write_pipeline = VK_NULL_HANDLE;
        }

        if (per_local_size.write_shader_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context.device(), per_local_size.write_shader_module, nullptr);
            per_local_size.write_shader_module = VK_NULL_HANDLE;
        }
    }

    resources.clear();
}

bool create_shared_write_resources(VulkanContext& context, VkDeviceSize buffer_size, SharedWriteResources& out) {
    if (!create_buffer_resource(context.physical_device(), context.device(), buffer_size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out.device_storage)) {
        std::cerr << "Failed to create device-local storage buffer for local size sweep.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out.upload_staging)) {
        std::cerr << "Failed to create upload staging buffer for local size sweep.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out.readback_staging)) {
        std::cerr << "Failed to create readback staging buffer for local size sweep.\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings, out.descriptor_set_layout)) {
        std::cerr << "Failed to create descriptor set layout for local size sweep.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U}};
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out.descriptor_pool)) {
        std::cerr << "Failed to create descriptor pool for local size sweep.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out.descriptor_pool, out.descriptor_set_layout,
                                                     out.descriptor_set)) {
        std::cerr << "Failed to allocate descriptor set for local size sweep.\n";
        return false;
    }

    const VkDescriptorBufferInfo buffer_info{out.device_storage.buffer, 0U, out.device_storage.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out.descriptor_set,
        {VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = buffer_info}});

    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out.descriptor_set_layout},
                                                    out.pipeline_layout)) {
        std::cerr << "Failed to create write pipeline layout for local size sweep.\n";
        return false;
    }

    return true;
}

bool create_shared_noop_resources(VulkanContext& context, SharedNoopResources& out) {
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {}, out.pipeline_layout)) {
        std::cerr << "Failed to create no-op pipeline layout for local size sweep.\n";
        return false;
    }

    return true;
}

bool create_local_size_pipelines(VulkanContext& context, const std::vector<uint32_t>& local_sizes,
                                 const SharedWriteResources& write_resources, const SharedNoopResources& noop_resources,
                                 bool include_noop_variant, std::vector<LocalSizePipelineResources>& out_resources) {
    out_resources.clear();
    out_resources.reserve(local_sizes.size());

    for (const uint32_t local_size_x : local_sizes) {
        LocalSizePipelineResources resources{};
        resources.local_size_x = local_size_x;
        const auto destroy_uncommitted = [&]() {
            if (resources.noop_pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(context.device(), resources.noop_pipeline, nullptr);
                resources.noop_pipeline = VK_NULL_HANDLE;
            }

            if (resources.noop_shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(context.device(), resources.noop_shader_module, nullptr);
                resources.noop_shader_module = VK_NULL_HANDLE;
            }

            if (resources.write_pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(context.device(), resources.write_pipeline, nullptr);
                resources.write_pipeline = VK_NULL_HANDLE;
            }

            if (resources.write_shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(context.device(), resources.write_shader_module, nullptr);
                resources.write_shader_module = VK_NULL_HANDLE;
            }
        };

        const std::string write_shader_path =
            VulkanComputeUtils::resolve_shader_path({}, write_shader_name(local_size_x));
        if (write_shader_path.empty()) {
            std::cerr << "Could not locate write shader for local_size_x=" << local_size_x << "\n";
            return false;
        }

        if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), write_shader_path,
                                                              resources.write_shader_module)) {
            std::cerr << "Failed to load write shader module: " << write_shader_path << "\n";
            destroy_uncommitted();
            return false;
        }

        if (!VulkanComputeUtils::create_compute_pipeline(context.device(), resources.write_shader_module,
                                                         write_resources.pipeline_layout, "main",
                                                         resources.write_pipeline)) {
            std::cerr << "Failed to create write pipeline for local_size_x=" << local_size_x << "\n";
            destroy_uncommitted();
            return false;
        }

        if (include_noop_variant) {
            const std::string noop_shader_path =
                VulkanComputeUtils::resolve_shader_path({}, noop_shader_name(local_size_x));
            if (noop_shader_path.empty()) {
                std::cerr << "Could not locate no-op shader for local_size_x=" << local_size_x << "\n";
                destroy_uncommitted();
                return false;
            }

            if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), noop_shader_path,
                                                                  resources.noop_shader_module)) {
                std::cerr << "Failed to load no-op shader module: " << noop_shader_path << "\n";
                destroy_uncommitted();
                return false;
            }

            if (!VulkanComputeUtils::create_compute_pipeline(context.device(), resources.noop_shader_module,
                                                             noop_resources.pipeline_layout, "main",
                                                             resources.noop_pipeline)) {
                std::cerr << "Failed to create no-op pipeline for local_size_x=" << local_size_x << "\n";
                destroy_uncommitted();
                return false;
            }
        }

        out_resources.push_back(resources);
    }

    return true;
}

double run_upload_stage(VulkanContext& context, const SharedWriteResources& resources, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, resources.upload_staging.buffer, resources.device_storage.buffer, 1U,
                        &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                resources.device_storage.buffer, bytes);
    });
}

double run_dispatch_write_stage(VulkanContext& context, const SharedWriteResources& resources, VkPipeline pipeline,
                                VkDeviceSize bytes, uint32_t group_count_x, uint32_t dispatch_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        for (uint32_t dispatch_index = 0; dispatch_index < dispatch_count; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        }

        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer,
                                                                          resources.device_storage.buffer, bytes);
    });
}

double run_dispatch_noop_stage(VulkanContext& context, VkPipeline pipeline, uint32_t group_count_x,
                               uint32_t dispatch_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        for (uint32_t dispatch_index = 0; dispatch_index < dispatch_count; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        }
    });
}

double run_readback_stage(VulkanContext& context, const SharedWriteResources& resources, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, resources.device_storage.buffer, resources.readback_staging.buffer, 1U,
                        &copy_region);
    });
}

bool validate_write_result(const float* data, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        if (data[index] != static_cast<float>(index)) {
            return false;
        }
    }

    return true;
}

bool validate_noop_result(const float* data, uint32_t count, float sentinel_value) {
    for (uint32_t index = 0; index < count; ++index) {
        if (data[index] != sentinel_value) {
            return false;
        }
    }

    return true;
}

} // namespace

LocalSizeSweepExperimentOutput run_local_size_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                               const LocalSizeSweepExperimentConfig& config) {
    LocalSizeSweepExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "Local size sweep experiment requires GPU timestamp support.\n";
        return output;
    }

    if (config.dispatch_count == 0U) {
        std::cerr << "Local size sweep requires dispatch_count > 0.\n";
        return output;
    }

    const auto max_elements_u64 = static_cast<uint64_t>(config.max_buffer_bytes / sizeof(float));
    if (max_elements_u64 == 0U) {
        std::cerr << "Scratch buffer too small for local size sweep experiment.\n";
        return output;
    }

    const uint32_t max_elements =
        static_cast<uint32_t>(std::min<uint64_t>(max_elements_u64, std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> base_problem_sizes = make_base_problem_sizes(max_elements);
    if (base_problem_sizes.empty()) {
        std::cerr << "Scratch buffer is too small. Local size sweep requires at least 2^14 float elements.\n";
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    std::vector<uint32_t> legal_local_sizes;
    legal_local_sizes.reserve(kLocalSizeCandidates.size());
    for (const uint32_t local_size_x : kLocalSizeCandidates) {
        if (is_local_size_candidate_legal(device_properties, local_size_x)) {
            legal_local_sizes.push_back(local_size_x);
        } else {
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Skipping illegal local_size_x=" << local_size_x
                          << " (device limits).\n";
            }
        }
    }

    if (legal_local_sizes.empty()) {
        std::cerr << "No legal local_size_x candidates available for this device.\n";
        return output;
    }

    const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(base_problem_sizes.back()) * sizeof(float);
    SharedWriteResources write_resources{};
    if (!create_shared_write_resources(context, buffer_size, write_resources)) {
        destroy_shared_write_resources(context, write_resources);
        return output;
    }

    SharedNoopResources noop_resources{};
    if (config.include_noop_variant && !create_shared_noop_resources(context, noop_resources)) {
        destroy_shared_noop_resources(context, noop_resources);
        destroy_shared_write_resources(context, write_resources);
        return output;
    }

    std::vector<LocalSizePipelineResources> pipeline_resources;
    if (!create_local_size_pipelines(context, legal_local_sizes, write_resources, noop_resources,
                                     config.include_noop_variant, pipeline_resources)) {
        destroy_local_size_pipelines(context, pipeline_resources);
        destroy_shared_noop_resources(context, noop_resources);
        destroy_shared_write_resources(context, write_resources);
        return output;
    }

    void* upload_mapped_data = nullptr;
    const VkResult upload_map_result = vkMapMemory(context.device(), write_resources.upload_staging.memory, 0U,
                                                   write_resources.upload_staging.size, 0U, &upload_mapped_data);
    if (upload_map_result != VK_SUCCESS || upload_mapped_data == nullptr) {
        std::cerr << "vkMapMemory failed for upload staging buffer with error code " << upload_map_result << "\n";
        destroy_local_size_pipelines(context, pipeline_resources);
        destroy_shared_noop_resources(context, noop_resources);
        destroy_shared_write_resources(context, write_resources);
        return output;
    }

    void* readback_mapped_data = nullptr;
    const VkResult readback_map_result = vkMapMemory(context.device(), write_resources.readback_staging.memory, 0U,
                                                     write_resources.readback_staging.size, 0U, &readback_mapped_data);
    if (readback_map_result != VK_SUCCESS || readback_mapped_data == nullptr) {
        std::cerr << "vkMapMemory failed for readback staging buffer with error code " << readback_map_result << "\n";
        vkUnmapMemory(context.device(), write_resources.upload_staging.memory);
        destroy_local_size_pipelines(context, pipeline_resources);
        destroy_shared_noop_resources(context, noop_resources);
        destroy_shared_write_resources(context, write_resources);
        return output;
    }

    auto* upload_values = static_cast<float*>(upload_mapped_data);
    auto* readback_values = static_cast<float*>(readback_mapped_data);

    std::size_t total_case_count = 0U;
    const std::size_t variant_multiplier = config.include_noop_variant ? 2U : 1U;
    for (const LocalSizePipelineResources& per_local_size : pipeline_resources) {
        const std::vector<uint32_t> problem_sizes_for_local_size = filter_problem_sizes_for_local_size(
            base_problem_sizes, per_local_size.local_size_x, device_properties.limits.maxComputeWorkGroupCount[0]);
        total_case_count += problem_sizes_for_local_size.size() * variant_multiplier;
    }

    std::size_t completed_case_count = 0U;
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Starting run with local_sizes=" << pipeline_resources.size()
                  << ", base_problem_sizes=" << base_problem_sizes.size()
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << ", dispatch_count=" << config.dispatch_count
                  << ", include_noop_variant=" << (config.include_noop_variant ? "true" : "false") << "\n";
    }

    for (const LocalSizePipelineResources& per_local_size : pipeline_resources) {
        const uint32_t local_size_x = per_local_size.local_size_x;
        const std::vector<uint32_t> problem_sizes_for_local_size = filter_problem_sizes_for_local_size(
            base_problem_sizes, local_size_x, device_properties.limits.maxComputeWorkGroupCount[0]);
        if (problem_sizes_for_local_size.empty()) {
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Skipping local_size_x=" << local_size_x
                          << " (no legal problem sizes under maxComputeWorkGroupCount[0]).\n";
            }
            continue;
        }

        for (const uint32_t problem_size : problem_sizes_for_local_size) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(problem_size) * sizeof(float);
            const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, local_size_x);

            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Case " << (completed_case_count + 1U) << "/" << total_case_count
                          << ": variant=contiguous_write"
                          << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                          << ", group_count_x=" << group_count_x << "\n";
            }
            std::vector<double> write_samples;
            write_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

            for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                std::fill_n(upload_values, problem_size, kWriteSentinel);

                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_write_stage(context, write_resources, per_local_size.write_pipeline, bytes,
                                             group_count_x, config.dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);

                if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms)) {
                    std::cerr << "Warmup produced non-finite timing value for local_size_x=" << local_size_x
                              << ", variant=contiguous_write.\n";
                }

                if (verbose_progress) {
                    const bool warmup_ok =
                        std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                    std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                              << runner.warmup_iterations() << " variant=contiguous_write"
                              << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", correctness=" << (warmup_ok ? "pass" : "fail")
                              << "\n";
                }
            }

            for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                std::fill_n(upload_values, problem_size, kWriteSentinel);

                const auto start = std::chrono::high_resolution_clock::now();
                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_write_stage(context, write_resources, per_local_size.write_pipeline, bytes,
                                             group_count_x, config.dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);
                const auto end = std::chrono::high_resolution_clock::now();

                const bool upload_ok = std::isfinite(upload_ms);
                const bool dispatch_ok = std::isfinite(dispatch_ms);
                const bool readback_ok = std::isfinite(readback_ms);
                const bool data_ok = validate_write_result(readback_values, problem_size);
                const bool correctness = upload_ok && dispatch_ok && readback_ok && data_ok;

                std::string notes;
                append_note(notes, "local_size_x=" + std::to_string(local_size_x));
                append_note(notes, "group_count_x=" + std::to_string(group_count_x));
                if (!upload_ok) {
                    append_note(notes, "upload_ms_non_finite");
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

                const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;
                write_samples.push_back(dispatch_ms);
                if (verbose_progress) {
                    std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/"
                              << runner.timed_iterations() << " variant=contiguous_write"
                              << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                              << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                }
                output.rows.push_back(BenchmarkMeasurementRow{
                    .experiment_id = kExperimentId,
                    .variant = "contiguous_write_ls" + std::to_string(local_size_x),
                    .problem_size = problem_size,
                    .dispatch_count = config.dispatch_count,
                    .iteration = iteration,
                    .gpu_ms = dispatch_ms,
                    .end_to_end_ms = end_to_end_ms.count(),
                    .throughput =
                        compute_throughput_elements_per_second(problem_size, config.dispatch_count, dispatch_ms),
                    .gbps = compute_effective_gbps(problem_size, config.dispatch_count,
                                                   static_cast<uint32_t>(sizeof(float)), dispatch_ms),
                    .correctness_pass = correctness,
                    .notes = notes,
                });
                output.all_points_correct = output.all_points_correct && correctness;
            }

            const BenchmarkResult write_summary = BenchmarkRunner::summarize_samples(
                build_case_name("contiguous_write", local_size_x, problem_size, config.dispatch_count), write_samples);
            output.summary_results.push_back(write_summary);
            ++completed_case_count;
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Completed case " << completed_case_count << "/"
                          << total_case_count << ": variant=contiguous_write, local_size_x=" << local_size_x
                          << ", problem_size=" << problem_size << ", samples=" << write_summary.sample_count
                          << ", median_gpu_ms=" << write_summary.median_ms << "\n";
            }

            if (!config.include_noop_variant) {
                continue;
            }

            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Case " << (completed_case_count + 1U) << "/" << total_case_count
                          << ": variant=noop"
                          << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                          << ", group_count_x=" << group_count_x << "\n";
            }
            std::vector<double> noop_samples;
            noop_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

            for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                std::fill_n(upload_values, problem_size, kNoopSentinel);

                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms = run_dispatch_noop_stage(context, per_local_size.noop_pipeline, group_count_x,
                                                                   config.dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);

                if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms)) {
                    std::cerr << "Warmup produced non-finite timing value for local_size_x=" << local_size_x
                              << ", variant=noop.\n";
                }

                if (verbose_progress) {
                    const bool warmup_ok =
                        std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                    std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                              << runner.warmup_iterations() << " variant=noop"
                              << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", correctness=" << (warmup_ok ? "pass" : "fail")
                              << "\n";
                }
            }

            for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                std::fill_n(upload_values, problem_size, kNoopSentinel);

                const auto start = std::chrono::high_resolution_clock::now();
                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms = run_dispatch_noop_stage(context, per_local_size.noop_pipeline, group_count_x,
                                                                   config.dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);
                const auto end = std::chrono::high_resolution_clock::now();

                const bool upload_ok = std::isfinite(upload_ms);
                const bool dispatch_ok = std::isfinite(dispatch_ms);
                const bool readback_ok = std::isfinite(readback_ms);
                const bool data_ok = validate_noop_result(readback_values, problem_size, kNoopSentinel);
                const bool correctness = upload_ok && dispatch_ok && readback_ok && data_ok;

                std::string notes;
                append_note(notes, "local_size_x=" + std::to_string(local_size_x));
                append_note(notes, "group_count_x=" + std::to_string(group_count_x));
                append_note(notes, "no_payload_bytes");
                if (!upload_ok) {
                    append_note(notes, "upload_ms_non_finite");
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

                const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;
                noop_samples.push_back(dispatch_ms);
                if (verbose_progress) {
                    std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/"
                              << runner.timed_iterations() << " variant=noop"
                              << ", local_size_x=" << local_size_x << ", problem_size=" << problem_size
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                              << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                }
                output.rows.push_back(BenchmarkMeasurementRow{
                    .experiment_id = kExperimentId,
                    .variant = "noop_ls" + std::to_string(local_size_x),
                    .problem_size = problem_size,
                    .dispatch_count = config.dispatch_count,
                    .iteration = iteration,
                    .gpu_ms = dispatch_ms,
                    .end_to_end_ms = end_to_end_ms.count(),
                    .throughput = 0.0,
                    .gbps = 0.0,
                    .correctness_pass = correctness,
                    .notes = notes,
                });
                output.all_points_correct = output.all_points_correct && correctness;
            }

            const BenchmarkResult noop_summary = BenchmarkRunner::summarize_samples(
                build_case_name("noop", local_size_x, problem_size, config.dispatch_count), noop_samples);
            output.summary_results.push_back(noop_summary);
            ++completed_case_count;
            if (verbose_progress) {
                std::cout << "[" << kExperimentId << "] Completed case " << completed_case_count << "/"
                          << total_case_count << ": variant=noop, local_size_x=" << local_size_x
                          << ", problem_size=" << problem_size << ", samples=" << noop_summary.sample_count
                          << ", median_gpu_ms=" << noop_summary.median_ms << "\n";
            }
        }
    }

    vkUnmapMemory(context.device(), write_resources.readback_staging.memory);
    vkUnmapMemory(context.device(), write_resources.upload_staging.memory);
    destroy_local_size_pipelines(context, pipeline_resources);
    destroy_shared_noop_resources(context, noop_resources);
    destroy_shared_write_resources(context, write_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
