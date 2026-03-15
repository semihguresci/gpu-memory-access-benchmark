#include "experiments/global_id_mapping_variants_experiment.hpp"

#include "utils/buffer_utils.hpp"
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

constexpr const char* kExperimentId = "05_global_id_mapping_variants";
constexpr uint32_t kLocalSizeX = 256U;
constexpr uint32_t kMinProblemPower = 10U;
constexpr uint32_t kMaxProblemPower = 24U;
constexpr uint32_t kFixedOffsetElements = 17U;
constexpr float kOutputSentinel = -65432.0F;
constexpr std::array<uint32_t, 8> kDispatchCounts = {1U, 4U, 16U, 64U, 128U, 256U, 512U, 1024U};

enum class MappingMode : uint8_t {
    kDirect = 0U,
    kOffset = 1U,
    kGridStride = 2U,
};

struct MappingVariant {
    const char* name = "";
    MappingMode mode = MappingMode::kDirect;
};

constexpr std::array<MappingVariant, 3> kMappingVariants = {
    MappingVariant{.name = "direct", .mode = MappingMode::kDirect},
    MappingVariant{.name = "offset", .mode = MappingMode::kOffset},
    MappingVariant{.name = "grid_stride", .mode = MappingMode::kGridStride},
};

struct ExperimentPushConstants {
    uint32_t element_count = 0U;
    uint32_t mapping_mode = 0U;
    uint32_t fixed_offset = 0U;
    uint32_t total_invocations = 0U;
};

static_assert(sizeof(ExperimentPushConstants) == (sizeof(uint32_t) * 4U));

struct ExperimentBufferResources {
    BufferResource src_device{};
    BufferResource dst_device{};
    BufferResource staging{};
};

struct PipelineResources {
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

std::string build_case_name(uint32_t problem_size, uint32_t dispatch_count, const char* variant_name) {
    return std::string(kExperimentId) + "_" + variant_name + "_size_" + std::to_string(problem_size) + "_dispatches_" +
           std::to_string(dispatch_count);
}

float source_pattern_value(uint32_t index) {
    return static_cast<float>(static_cast<int32_t>(index % 4096U) - 2048);
}

float expected_output_value(uint32_t index) {
    return source_pattern_value(index) + static_cast<float>(index & 7U);
}

void fill_source_pattern(float* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_sentinel(float* values, uint32_t element_count, float sentinel_value) {
    std::fill_n(values, element_count, sentinel_value);
}

bool validate_output_pattern(const float* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != expected_output_value(index)) {
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
    return (elements * 1000.0) / dispatch_gpu_ms;
}

double compute_effective_gbps(uint32_t problem_size, uint32_t dispatch_count, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes_per_dispatch = static_cast<double>(problem_size) * static_cast<double>(sizeof(float) * 2U);
    const double bytes = bytes_per_dispatch * static_cast<double>(dispatch_count);
    return bytes / (dispatch_gpu_ms * 1.0e6);
}

std::vector<uint32_t> make_problem_sizes(uint32_t max_elements, uint32_t max_group_count_x) {
    std::vector<uint32_t> sizes;
    sizes.reserve(kMaxProblemPower - kMinProblemPower + 1U);

    for (uint32_t power = kMinProblemPower; power <= kMaxProblemPower; ++power) {
        const uint64_t elements_u64 = 1ULL << power;
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

uint32_t compute_total_invocations(uint32_t group_count_x) {
    const uint64_t total_invocations = static_cast<uint64_t>(group_count_x) * static_cast<uint64_t>(kLocalSizeX);
    if (total_invocations > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<uint32_t>::max();
    }

    return static_cast<uint32_t>(total_invocations);
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
        std::cerr << "Failed to create source buffer for global ID mapping variants experiment.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), max_buffer_size,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.dst_device)) {
        std::cerr << "Failed to create destination buffer for global ID mapping variants experiment.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), max_buffer_size,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging)) {
        std::cerr << "Failed to create staging buffer for global ID mapping variants experiment.\n";
        return false;
    }

    return true;
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResource& src_device,
                               const BufferResource& dst_device, PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load global ID mapping variants shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create global ID mapping variants descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create global ID mapping variants descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate global ID mapping variants descriptor set.\n";
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
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(ExperimentPushConstants)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create global ID mapping variants pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create global ID mapping variants compute pipeline.\n";
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

double run_dispatch_stage(VulkanContext& context, const PipelineResources& resources, const BufferResource& dst_buffer,
                          VkDeviceSize bytes, uint32_t group_count_x, uint32_t dispatch_count,
                          const ExperimentPushConstants& push_constants) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(push_constants), &push_constants);

        for (uint32_t dispatch_index = 0U; dispatch_index < dispatch_count; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        }

        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, dst_buffer.buffer, bytes);
    });
}

double run_readback_stage(VulkanContext& context, const BufferResource& src_buffer,
                          const BufferResource& staging_buffer, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, src_buffer.buffer, staging_buffer.buffer, 1U, &copy_region);
    });
}

} // namespace

GlobalIdMappingVariantsExperimentOutput
run_global_id_mapping_variants_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                          const GlobalIdMappingVariantsExperimentConfig& config) {
    GlobalIdMappingVariantsExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "Global ID mapping variants experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "05_global_id_mapping_variants.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for global ID mapping variants experiment.\n";
        return output;
    }

    const auto buffer_limited_elements_u64 = static_cast<uint64_t>(config.max_buffer_bytes / sizeof(float));
    if (buffer_limited_elements_u64 == 0U) {
        std::cerr << "Scratch buffer too small for global ID mapping variants experiment.\n";
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint64_t dispatch_limited_elements_u64 =
        static_cast<uint64_t>(device_properties.limits.maxComputeWorkGroupCount[0]) * kLocalSizeX;
    const uint64_t effective_max_elements_u64 = std::min({buffer_limited_elements_u64, dispatch_limited_elements_u64,
                                                          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (effective_max_elements_u64 == 0U) {
        std::cerr << "No legal problem sizes available for global ID mapping variants experiment.\n";
        return output;
    }

    if (effective_max_elements_u64 < buffer_limited_elements_u64) {
        std::cerr << "Global ID mapping variants sweep clamped by maxComputeWorkGroupCount[0].\n";
    }

    const auto max_elements = static_cast<uint32_t>(effective_max_elements_u64);
    const std::vector<uint32_t> problem_sizes =
        make_problem_sizes(max_elements, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (problem_sizes.empty()) {
        std::cerr << "Scratch buffer is too small. Global ID mapping variants requires at least 2^10 float elements.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] Starting run with problem_sizes=" << problem_sizes.size()
                  << ", dispatch_counts=" << kDispatchCounts.size() << ", variants=" << kMappingVariants.size()
                  << ", local_size_x=" << kLocalSizeX << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const VkDeviceSize max_buffer_size = static_cast<VkDeviceSize>(problem_sizes.back()) * sizeof(float);
    ExperimentBufferResources buffers{};
    if (!create_experiment_buffer_resources(context, max_buffer_size, buffers)) {
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, buffers.src_device, buffers.dst_device, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    void* mapped_staging = nullptr;
    const VkResult map_result =
        vkMapMemory(context.device(), buffers.staging.memory, 0U, buffers.staging.size, 0U, &mapped_staging);
    if (map_result != VK_SUCCESS || mapped_staging == nullptr) {
        std::cerr << "vkMapMemory failed for global ID mapping variants staging buffer with error code " << map_result
                  << "\n";
        destroy_pipeline_resources(context, pipeline_resources);
        destroy_experiment_buffer_resources(context, buffers);
        return output;
    }

    auto* staging_values = static_cast<float*>(mapped_staging);
    const std::size_t total_case_count = problem_sizes.size() * kDispatchCounts.size() * kMappingVariants.size();
    std::size_t completed_case_count = 0U;

    for (const uint32_t problem_size : problem_sizes) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(problem_size) * sizeof(float);
        const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, kLocalSizeX);
        const uint32_t total_invocations = compute_total_invocations(group_count_x);

        for (const MappingVariant variant : kMappingVariants) {
            for (const uint32_t dispatch_count : kDispatchCounts) {
                if (verbose_progress) {
                    std::cout << "[" << kExperimentId << "] Case " << (completed_case_count + 1U) << "/"
                              << total_case_count << ": variant=" << variant.name << ", problem_size=" << problem_size
                              << ", dispatch_count=" << dispatch_count << ", group_count_x=" << group_count_x
                              << ", total_invocations=" << total_invocations << "\n";
                }

                std::vector<double> dispatch_samples;
                dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

                for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
                    fill_source_pattern(staging_values, problem_size);
                    const double upload_src_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_sentinel(staging_values, problem_size, kOutputSentinel);
                    const double upload_dst_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    const ExperimentPushConstants push_constants{
                        .element_count = problem_size,
                        .mapping_mode = static_cast<uint32_t>(variant.mode),
                        .fixed_offset = kFixedOffsetElements,
                        .total_invocations = total_invocations,
                    };
                    const double dispatch_ms = run_dispatch_stage(context, pipeline_resources, buffers.dst_device,
                                                                  bytes, group_count_x, dispatch_count, push_constants);
                    const double readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);

                    if (!std::isfinite(upload_src_ms) || !std::isfinite(upload_dst_ms) || !std::isfinite(dispatch_ms) ||
                        !std::isfinite(readback_ms)) {
                        std::cerr
                            << "Warmup produced non-finite timing value in global ID mapping variants experiment.\n";
                    }

                    if (verbose_progress) {
                        const bool warmup_ok = std::isfinite(upload_src_ms) && std::isfinite(upload_dst_ms) &&
                                               std::isfinite(dispatch_ms) && std::isfinite(readback_ms);
                        std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/"
                                  << runner.warmup_iterations() << " variant=" << variant.name
                                  << ", problem_size=" << problem_size << ", dispatch_count=" << dispatch_count
                                  << ", upload_src_ms=" << upload_src_ms << ", upload_dst_ms=" << upload_dst_ms
                                  << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                                  << ", correctness=" << (warmup_ok ? "pass" : "fail") << "\n";
                    }
                }

                for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
                    const auto start = std::chrono::high_resolution_clock::now();

                    fill_source_pattern(staging_values, problem_size);
                    const double upload_src_ms = run_upload_stage(context, buffers.staging, buffers.src_device, bytes);
                    fill_sentinel(staging_values, problem_size, kOutputSentinel);
                    const double upload_dst_ms = run_upload_stage(context, buffers.staging, buffers.dst_device, bytes);
                    const ExperimentPushConstants push_constants{
                        .element_count = problem_size,
                        .mapping_mode = static_cast<uint32_t>(variant.mode),
                        .fixed_offset = kFixedOffsetElements,
                        .total_invocations = total_invocations,
                    };
                    const double dispatch_ms = run_dispatch_stage(context, pipeline_resources, buffers.dst_device,
                                                                  bytes, group_count_x, dispatch_count, push_constants);
                    const double readback_ms = run_readback_stage(context, buffers.dst_device, buffers.staging, bytes);

                    const auto end = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

                    const bool upload_src_ok = std::isfinite(upload_src_ms);
                    const bool upload_dst_ok = std::isfinite(upload_dst_ms);
                    const bool dispatch_ok = std::isfinite(dispatch_ms);
                    const bool readback_ok = std::isfinite(readback_ms);
                    const bool data_ok = validate_output_pattern(staging_values, problem_size);

                    std::string notes;
                    append_note(notes, "local_size_x=" + std::to_string(kLocalSizeX));
                    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
                    append_note(notes, "total_invocations=" + std::to_string(total_invocations));
                    append_note(notes, "mapping=" + std::string(variant.name));
                    if (variant.mode == MappingMode::kOffset) {
                        append_note(notes, "fixed_offset=" + std::to_string(kFixedOffsetElements));
                    }
                    if (!upload_src_ok) {
                        append_note(notes, "upload_src_ms_non_finite");
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

                    const bool correctness = upload_src_ok && upload_dst_ok && dispatch_ok && readback_ok && data_ok;
                    dispatch_samples.push_back(dispatch_ms);
                    if (verbose_progress) {
                        std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/"
                                  << runner.timed_iterations() << " variant=" << variant.name
                                  << ", problem_size=" << problem_size << ", dispatch_count=" << dispatch_count
                                  << ", upload_src_ms=" << upload_src_ms << ", upload_dst_ms=" << upload_dst_ms
                                  << ", dispatch_ms=" << dispatch_ms << ", readback_ms=" << readback_ms
                                  << ", end_to_end_ms=" << end_to_end_ms.count()
                                  << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
                    }
                    output.rows.push_back(BenchmarkMeasurementRow{
                        .experiment_id = kExperimentId,
                        .variant = variant.name,
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

                const BenchmarkResult summary = BenchmarkRunner::summarize_samples(
                    build_case_name(problem_size, dispatch_count, variant.name), dispatch_samples);
                output.summary_results.push_back(summary);
                ++completed_case_count;
                if (verbose_progress) {
                    std::cout << "[" << kExperimentId << "] Completed case " << completed_case_count << "/"
                              << total_case_count << ": variant=" << variant.name << ", problem_size=" << problem_size
                              << ", dispatch_count=" << dispatch_count << ", samples=" << summary.sample_count
                              << ", median_gpu_ms=" << summary.median_ms << "\n";
                }
            }
        }
    }

    vkUnmapMemory(context.device(), buffers.staging.memory);
    destroy_pipeline_resources(context, pipeline_resources);
    destroy_experiment_buffer_resources(context, buffers);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }
    return output;
}
