#include "experiments/dispatch_basics_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLocalSizeX = 64;
constexpr uint32_t kMinProblemPower = 10;
constexpr uint32_t kMaxProblemPower = 24;
constexpr float kWriteSentinel = -1.0F;
constexpr float kNoopSentinel = -2.0F;
constexpr std::array<uint32_t, 8> kDispatchCounts = {1, 4, 16, 64, 128, 256, 512, 1024};

struct WritePipelineResources {
    BufferResource device_storage{};
    BufferResource upload_staging{};
    BufferResource readback_staging{};
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct NoopPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

std::vector<uint32_t> make_problem_sizes(uint32_t max_elements) {
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

void destroy_write_pipeline_resources(VulkanContext& context, WritePipelineResources& resources) {
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

    destroy_buffer_resource(context.device(), resources.readback_staging);
    destroy_buffer_resource(context.device(), resources.upload_staging);
    destroy_buffer_resource(context.device(), resources.device_storage);
}

void destroy_noop_pipeline_resources(VulkanContext& context, NoopPipelineResources& resources) {
    if (resources.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device(), resources.pipeline, nullptr);
        resources.pipeline = VK_NULL_HANDLE;
    }

    if (resources.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device(), resources.pipeline_layout, nullptr);
        resources.pipeline_layout = VK_NULL_HANDLE;
    }

    if (resources.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device(), resources.shader_module, nullptr);
        resources.shader_module = VK_NULL_HANDLE;
    }
}

bool create_write_pipeline_resources(VulkanContext& context, const std::string& shader_path, VkDeviceSize buffer_size,
                                     WritePipelineResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), buffer_size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.device_storage)) {
        std::cerr << "Failed to create device-local storage buffer for dispatch basics experiment.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.upload_staging)) {
        std::cerr << "Failed to create upload staging buffer for dispatch basics experiment.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.readback_staging)) {
        std::cerr << "Failed to create readback staging buffer for dispatch basics experiment.\n";
        return false;
    }

    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load write shader module from: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};

    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create write descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U}};
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create write descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate write descriptor set.\n";
        return false;
    }

    const VkDescriptorBufferInfo buffer_info{out_resources.device_storage.buffer, 0U,
                                             out_resources.device_storage.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), out_resources.descriptor_set,
        {VulkanComputeUtils::DescriptorBufferBindingUpdate{
            .binding = 0U, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = buffer_info}});

    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    out_resources.pipeline_layout)) {
        std::cerr << "Failed to create write pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create write compute pipeline.\n";
        return false;
    }

    return true;
}

bool create_noop_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                                    NoopPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load no-op shader module from: " << shader_path << "\n";
        return false;
    }

    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {}, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create no-op pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create no-op compute pipeline.\n";
        return false;
    }

    return true;
}

double run_upload_stage(VulkanContext& context, const WritePipelineResources& resources, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, resources.upload_staging.buffer, resources.device_storage.buffer, 1U,
                        &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                resources.device_storage.buffer, bytes);
    });
}

double run_dispatch_write_stage(VulkanContext& context, const WritePipelineResources& resources, VkDeviceSize bytes,
                                uint32_t group_count_x, uint32_t dispatch_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        for (uint32_t dispatch_index = 0; dispatch_index < dispatch_count; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        }

        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer,
                                                                          resources.device_storage.buffer, bytes);
    });
}

double run_dispatch_noop_stage(VulkanContext& context, const NoopPipelineResources& resources, uint32_t group_count_x,
                               uint32_t dispatch_count) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        for (uint32_t dispatch_index = 0; dispatch_index < dispatch_count; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        }
    });
}

double run_readback_stage(VulkanContext& context, const WritePipelineResources& resources, VkDeviceSize bytes) {
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

double compute_throughput_elements_per_second(uint32_t problem_size, uint32_t dispatch_count, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double elements = static_cast<double>(problem_size) * static_cast<double>(dispatch_count);
    return elements * 1000.0 / dispatch_gpu_ms;
}

double compute_effective_gbps(uint32_t problem_size, uint32_t dispatch_count, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes = static_cast<double>(problem_size) * static_cast<double>(dispatch_count) * sizeof(float);
    return bytes / (dispatch_gpu_ms * 1.0e6);
}

std::string build_case_name(const std::string& variant, uint32_t problem_size, uint32_t dispatch_count) {
    return "01_dispatch_basics_" + variant + "_size_" + std::to_string(problem_size) + "_dispatches_" +
           std::to_string(dispatch_count);
}

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

} // namespace

DispatchBasicsExperimentOutput run_dispatch_basics_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                              const DispatchBasicsExperimentConfig& config) {
    DispatchBasicsExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "Dispatch basics experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string write_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.write_shader_path, "01_dispatch_basics_write.comp.spv");
    if (write_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for dispatch basics write variant.\n";
        return output;
    }
    if (verbose_progress) {
        std::cout << "[01_dispatch_basics] Write shader: " << write_shader_path << "\n";
    }

    std::string noop_shader_path;
    if (config.include_noop_variant) {
        noop_shader_path = VulkanComputeUtils::resolve_shader_path(config.noop_shader_path, "01_noop.comp.spv");
        if (noop_shader_path.empty()) {
            std::cerr << "Could not locate SPIR-V shader for dispatch basics no-op variant.\n";
            return output;
        }
        if (verbose_progress) {
            std::cout << "[01_dispatch_basics] No-op shader: " << noop_shader_path << "\n";
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    const uint64_t dispatch_limited_elements =
        static_cast<uint64_t>(device_properties.limits.maxComputeWorkGroupCount[0]) * kLocalSizeX;
    const uint64_t memory_limited_elements = static_cast<uint64_t>(config.max_buffer_bytes / sizeof(float));
    const uint64_t effective_max_elements = std::min(dispatch_limited_elements, memory_limited_elements);
    const auto max_elements =
        static_cast<uint32_t>(std::min<uint64_t>(effective_max_elements, std::numeric_limits<uint32_t>::max()));

    if (effective_max_elements < memory_limited_elements) {
        std::cerr << "Dispatch basics sweep clamped by maxComputeWorkGroupCount[0].\n";
    }

    const std::vector<uint32_t> problem_sizes = make_problem_sizes(max_elements);
    if (problem_sizes.empty()) {
        std::cerr << "Scratch buffer is too small. Dispatch basics requires at least 2^10 float elements.\n";
        return output;
    }
    const std::size_t variant_count = config.include_noop_variant ? 2U : 1U;
    const std::size_t total_case_count = problem_sizes.size() * kDispatchCounts.size() * variant_count;
    std::size_t completed_case_count = 0U;

    if (verbose_progress) {
        std::cout << "[01_dispatch_basics] Starting run with problem_sizes=" << problem_sizes.size()
                  << ", dispatch_counts=" << kDispatchCounts.size() << ", variants=" << variant_count
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations()
                  << ", max_buffer_bytes=" << config.max_buffer_bytes << "\n";
    }

    WritePipelineResources write_resources{};
    const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(problem_sizes.back()) * sizeof(float);
    if (!create_write_pipeline_resources(context, write_shader_path, buffer_size, write_resources)) {
        destroy_write_pipeline_resources(context, write_resources);
        return output;
    }

    NoopPipelineResources noop_resources{};
    if (config.include_noop_variant && !create_noop_pipeline_resources(context, noop_shader_path, noop_resources)) {
        destroy_noop_pipeline_resources(context, noop_resources);
        destroy_write_pipeline_resources(context, write_resources);
        return output;
    }

    void* upload_mapped_data = nullptr;
    const VkResult upload_map_result = vkMapMemory(context.device(), write_resources.upload_staging.memory, 0U,
                                                   write_resources.upload_staging.size, 0U, &upload_mapped_data);
    if (upload_map_result != VK_SUCCESS || upload_mapped_data == nullptr) {
        std::cerr << "vkMapMemory failed for upload staging buffer with error code " << upload_map_result << "\n";
        destroy_noop_pipeline_resources(context, noop_resources);
        destroy_write_pipeline_resources(context, write_resources);
        return output;
    }

    void* readback_mapped_data = nullptr;
    const VkResult readback_map_result = vkMapMemory(context.device(), write_resources.readback_staging.memory, 0U,
                                                     write_resources.readback_staging.size, 0U, &readback_mapped_data);
    if (readback_map_result != VK_SUCCESS || readback_mapped_data == nullptr) {
        std::cerr << "vkMapMemory failed for readback staging buffer with error code " << readback_map_result << "\n";
        vkUnmapMemory(context.device(), write_resources.upload_staging.memory);
        destroy_noop_pipeline_resources(context, noop_resources);
        destroy_write_pipeline_resources(context, write_resources);
        return output;
    }

    auto* upload_values = static_cast<float*>(upload_mapped_data);
    auto* readback_values = static_cast<float*>(readback_mapped_data);

    for (uint32_t problem_size : problem_sizes) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(problem_size) * sizeof(float);
        const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, kLocalSizeX);
        if (verbose_progress) {
            std::cout << "[01_dispatch_basics] Problem size=" << problem_size << " elements (" << bytes
                      << " bytes), group_count_x=" << group_count_x << "\n";
        }

        for (uint32_t dispatch_count : kDispatchCounts) {
            if (verbose_progress) {
                std::cout << "[01_dispatch_basics] Case " << (completed_case_count + 1U) << "/" << total_case_count
                          << ": variant=contiguous_write, dispatch_count=" << dispatch_count << "\n";
            }
            std::vector<double> write_samples;
            write_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

            for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                std::fill_n(upload_values, problem_size, kWriteSentinel);

                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_write_stage(context, write_resources, bytes, group_count_x, dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);

                if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms)) {
                    std::cerr << "Warmup produced non-finite timing value in dispatch basics write path.\n";
                }

                if (verbose_progress) {
                    const bool warmup_ok =
                        std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                    std::cout << "[01_dispatch_basics] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                              << " variant=contiguous_write, size=" << problem_size
                              << ", dispatch_count=" << dispatch_count << ", upload_ms=" << upload_ms
                              << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                              << ", correctness=" << (warmup_ok ? "pass" : "fail") << "\n";
                }
            }

            for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                std::fill_n(upload_values, problem_size, kWriteSentinel);

                const auto start = std::chrono::high_resolution_clock::now();
                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_write_stage(context, write_resources, bytes, group_count_x, dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);
                const auto end = std::chrono::high_resolution_clock::now();

                const bool upload_ok = std::isfinite(upload_ms);
                const bool dispatch_ok = std::isfinite(dispatch_ms);
                const bool readback_ok = std::isfinite(readback_ms);
                const bool data_ok = validate_write_result(readback_values, problem_size);
                const bool correctness = upload_ok && dispatch_ok && readback_ok && data_ok;

                std::string notes;
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
                    std::cout << "[01_dispatch_basics] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                              << " variant=contiguous_write, size=" << problem_size
                              << ", dispatch_count=" << dispatch_count << ", upload_ms=" << upload_ms
                              << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                              << ", end_to_end_ms=" << end_to_end_ms.count()
                              << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                }
                output.rows.push_back(BenchmarkMeasurementRow{
                    .experiment_id = "01_dispatch_basics",
                    .variant = "contiguous_write",
                    .problem_size = problem_size,
                    .dispatch_count = dispatch_count,
                    .iteration = iteration,
                    .gpu_ms = dispatch_ms,
                    .end_to_end_ms = end_to_end_ms.count(),
                    .throughput = compute_throughput_elements_per_second(problem_size, dispatch_count, dispatch_ms),
                    .gbps = compute_effective_gbps(problem_size, dispatch_count, dispatch_ms),
                    .correctness_pass = correctness,
                    .notes = notes,
                });
                output.all_points_correct = output.all_points_correct && correctness;
            }

            BenchmarkResult write_summary = BenchmarkRunner::summarize_samples(
                build_case_name("contiguous_write", problem_size, dispatch_count), write_samples);
            output.summary_results.push_back(write_summary);
            ++completed_case_count;
            if (verbose_progress) {
                std::cout << "[01_dispatch_basics] Completed case " << completed_case_count << "/" << total_case_count
                          << ": variant=contiguous_write, samples=" << write_summary.sample_count
                          << ", median_gpu_ms=" << write_summary.median_ms << "\n";
            }

            if (!config.include_noop_variant) {
                continue;
            }

            if (verbose_progress) {
                std::cout << "[01_dispatch_basics] Case " << (completed_case_count + 1U) << "/" << total_case_count
                          << ": variant=noop, dispatch_count=" << dispatch_count << "\n";
            }
            std::vector<double> noop_samples;
            noop_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

            for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                std::fill_n(upload_values, problem_size, kNoopSentinel);

                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_noop_stage(context, noop_resources, group_count_x, dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);

                if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms)) {
                    std::cerr << "Warmup produced non-finite timing value in dispatch basics no-op path.\n";
                }

                if (verbose_progress) {
                    const bool warmup_ok =
                        std::isfinite(upload_ms) && std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                    std::cout << "[01_dispatch_basics] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                              << " variant=noop, size=" << problem_size << ", dispatch_count=" << dispatch_count
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", correctness=" << (warmup_ok ? "pass" : "fail")
                              << "\n";
                }
            }

            for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                std::fill_n(upload_values, problem_size, kNoopSentinel);

                const auto start = std::chrono::high_resolution_clock::now();
                const double upload_ms = run_upload_stage(context, write_resources, bytes);
                const double dispatch_ms =
                    run_dispatch_noop_stage(context, noop_resources, group_count_x, dispatch_count);
                const double readback_ms = run_readback_stage(context, write_resources, bytes);
                const auto end = std::chrono::high_resolution_clock::now();

                const bool upload_ok = std::isfinite(upload_ms);
                const bool dispatch_ok = std::isfinite(dispatch_ms);
                const bool readback_ok = std::isfinite(readback_ms);
                const bool data_ok = validate_noop_result(readback_values, problem_size, kNoopSentinel);
                const bool correctness = upload_ok && dispatch_ok && readback_ok && data_ok;

                std::string notes;
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
                    std::cout << "[01_dispatch_basics] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                              << " variant=noop, size=" << problem_size << ", dispatch_count=" << dispatch_count
                              << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                              << ", readback_ms=" << readback_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                              << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                }
                output.rows.push_back(BenchmarkMeasurementRow{
                    .experiment_id = "01_dispatch_basics",
                    .variant = "noop",
                    .problem_size = problem_size,
                    .dispatch_count = dispatch_count,
                    .iteration = iteration,
                    .gpu_ms = dispatch_ms,
                    .end_to_end_ms = end_to_end_ms.count(),
                    .throughput = compute_throughput_elements_per_second(problem_size, dispatch_count, dispatch_ms),
                    .gbps = compute_effective_gbps(problem_size, dispatch_count, dispatch_ms),
                    .correctness_pass = correctness,
                    .notes = notes,
                });
                output.all_points_correct = output.all_points_correct && correctness;
            }

            BenchmarkResult noop_summary =
                BenchmarkRunner::summarize_samples(build_case_name("noop", problem_size, dispatch_count), noop_samples);
            output.summary_results.push_back(noop_summary);
            ++completed_case_count;
            if (verbose_progress) {
                std::cout << "[01_dispatch_basics] Completed case " << completed_case_count << "/" << total_case_count
                          << ": variant=noop, samples=" << noop_summary.sample_count
                          << ", median_gpu_ms=" << noop_summary.median_ms << "\n";
            }
        }
    }

    vkUnmapMemory(context.device(), write_resources.readback_staging.memory);
    vkUnmapMemory(context.device(), write_resources.upload_staging.memory);
    destroy_noop_pipeline_resources(context, noop_resources);
    destroy_write_pipeline_resources(context, write_resources);
    if (verbose_progress) {
        std::cout << "[01_dispatch_basics] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
