#include "experiments/coalesced_vs_strided_experiment.hpp"

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
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "11_coalesced_vs_strided";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kMaxStride = 64U;
constexpr uint32_t kSourceBaseValue = 0x01020304U;
constexpr uint32_t kSentinelValue = 0xA5A5A5A5U;
constexpr std::array<uint32_t, 7> kStrideValues = {1U, 2U, 4U, 8U, 16U, 32U, 64U};

struct CaseBufferResources {
    BufferResource src_buffer{};
    BufferResource dst_buffer{};
    void* src_mapped_ptr = nullptr;
    void* dst_mapped_ptr = nullptr;
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

std::string make_variant_name(uint32_t stride) {
    return "stride_" + std::to_string(stride);
}

std::string make_case_name(uint32_t logical_count, uint32_t stride) {
    return std::string(kExperimentId) + "_" + make_variant_name(stride) + "_elements_" + std::to_string(logical_count);
}

uint32_t source_pattern_value(uint32_t index) {
    return kSourceBaseValue + index;
}

void fill_source_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = source_pattern_value(index);
    }
}

void fill_sentinel_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, kSentinelValue);
}

bool validate_case_values(const uint32_t* src_values, const uint32_t* dst_values, uint32_t element_count,
                          uint32_t stride) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        const uint32_t expected_src = source_pattern_value(index);
        if (src_values[index] != expected_src) {
            return false;
        }

        if ((index % stride) == 0U) {
            if (dst_values[index] != (expected_src + 1U)) {
                return false;
            }
        } else if (dst_values[index] != kSentinelValue) {
            return false;
        }
    }

    return true;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t buffer_limited_count = max_buffer_bytes / (static_cast<uint64_t>(kMaxStride) * sizeof(uint32_t));
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count_u64 = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    return static_cast<uint32_t>(effective_count_u64);
}

VkDeviceSize compute_physical_span_bytes(uint32_t logical_count, uint32_t stride) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(stride) *
           static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_logical_bytes(uint32_t logical_count, uint32_t dispatch_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(dispatch_count) *
           static_cast<VkDeviceSize>(sizeof(uint32_t) * 2U);
}

double compute_effective_gbps(uint32_t logical_count, uint32_t dispatch_count, double dispatch_ms) {
    if (!std::isfinite(dispatch_ms) || dispatch_ms <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(compute_logical_bytes(logical_count, dispatch_count)) / (dispatch_ms * 1.0e6);
}

bool create_case_buffer_resources(VulkanContext& context, VkDeviceSize buffer_size,
                                  CaseBufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.src_buffer)) {
        std::cerr << "Failed to create source buffer for coalesced vs strided experiment.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create destination buffer for coalesced vs strided experiment.\n";
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "coalesced vs strided source buffer",
                           out_resources.src_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "coalesced vs strided destination buffer",
                           out_resources.dst_mapped_ptr)) {
        if (out_resources.src_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.src_buffer.memory);
            out_resources.src_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.dst_buffer);
        destroy_buffer_resource(context.device(), out_resources.src_buffer);
        return false;
    }

    return true;
}

void destroy_case_buffer_resources(VulkanContext& context, CaseBufferResources& resources) {
    if (resources.dst_mapped_ptr != nullptr && resources.dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.dst_buffer.memory);
        resources.dst_mapped_ptr = nullptr;
    }

    if (resources.src_mapped_ptr != nullptr && resources.src_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.src_buffer.memory);
        resources.src_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.dst_buffer);
    destroy_buffer_resource(context.device(), resources.src_buffer);
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load coalesced vs strided shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create coalesced vs strided descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create coalesced vs strided descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate coalesced vs strided descriptor set.\n";
        return false;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(uint32_t) * 2U)},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create coalesced vs strided pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create coalesced vs strided compute pipeline.\n";
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

void update_case_descriptor_set(VulkanContext& context, const PipelineResources& resources,
                                const CaseBufferResources& buffers) {
    const VkDescriptorBufferInfo src_info{
        buffers.src_buffer.buffer,
        0U,
        buffers.src_buffer.size,
    };
    const VkDescriptorBufferInfo dst_info{
        buffers.dst_buffer.buffer,
        0U,
        buffers.dst_buffer.size,
    };

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), resources.descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = src_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = dst_info,
                                                          },
                                                      });
}

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t logical_count,
                    uint32_t stride) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::array<uint32_t, 2> push_constants{logical_count, stride};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), push_constants.data());
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        VulkanComputeUtils::record_compute_write_to_host_read_barrier(command_buffer);
    });
}

void record_case_run(bool correctness_pass, bool dispatch_ok, double dispatch_ms, std::string& notes, uint32_t stride,
                     uint32_t logical_count, VkDeviceSize physical_span_bytes, VkDeviceSize allocated_span_bytes,
                     uint32_t group_count_x) {
    append_note(notes, "stride=" + std::to_string(stride));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_elements=" +
                           std::to_string(static_cast<unsigned long long>(physical_span_bytes / sizeof(uint32_t))));
    append_note(notes, "physical_span_bytes=" + std::to_string(static_cast<unsigned long long>(physical_span_bytes)));
    append_note(notes, "allocated_span_bytes=" + std::to_string(static_cast<unsigned long long>(allocated_span_bytes)));
    append_note(notes, "bytes_per_logical_element=" + std::to_string(sizeof(uint32_t) * 2U));
    append_note(notes, "validation_mode=exact_uint32");
    append_note(notes, std::string("access_pattern=") + (stride == 1U ? "coalesced" : "strided"));
    append_note(notes, "coalesced_baseline=" + std::string(stride == 1U ? "true" : "false"));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
    static_cast<void>(dispatch_ms);
}

template <typename ValidateFn, typename FillFn>
void run_stride_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
                     uint32_t logical_count, uint32_t stride, VkDeviceSize physical_span_bytes, bool verbose_progress,
                     FillFn fill_inputs, ValidateFn validate_outputs, CoalescedVsStridedExperimentOutput& output) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for stride " << stride << ".\n";
        return;
    }

    CaseBufferResources buffers{};
    if (!create_case_buffer_resources(context, physical_span_bytes, buffers)) {
        return;
    }

    update_case_descriptor_set(context, pipeline_resources, buffers);

    auto* src_values = static_cast<uint32_t*>(buffers.src_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for stride " << stride << ".\n";
        destroy_case_buffer_resources(context, buffers);
        return;
    }

    const std::string variant_name = make_variant_name(stride);
    const std::size_t timed_iterations = static_cast<std::size_t>(std::max(0, runner.timed_iterations()));
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(timed_iterations);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", physical_span_bytes=" << physical_span_bytes
                  << ", group_count_x=" << group_count_x << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_inputs(src_values, dst_values, logical_count, stride);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, stride);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(src_values, dst_values, logical_count, stride);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_elements=" << logical_count << ", stride=" << stride
                      << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_inputs(src_values, dst_values, logical_count, stride);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count, stride);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_outputs(src_values, dst_values, logical_count, stride);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_run(correctness_pass, dispatch_ok, dispatch_ms, notes, stride, logical_count, physical_span_bytes,
                        physical_span_bytes, group_count_x);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        append_note(notes, "gpu_time_scope=dispatch_and_host_visibility");
        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = logical_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(logical_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(logical_count, kDispatchCount, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(logical_count, stride), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    destroy_case_buffer_resources(context, buffers);
}

} // namespace

CoalescedVsStridedExperimentOutput
run_coalesced_vs_strided_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                    const CoalescedVsStridedExperimentConfig& config) {
    CoalescedVsStridedExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "coalesced vs strided experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "11_coalesced_vs_strided.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for coalesced vs strided experiment.\n";
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for coalesced vs strided experiment.\n";
        return output;
    }

    if (verbose_progress) {
        const VkDeviceSize max_physical_span_bytes = compute_physical_span_bytes(logical_count, kMaxStride);
        std::cout << "[" << kExperimentId << "] Shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_elements=" << logical_count
                  << ", max_physical_span_bytes=" << max_physical_span_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        return output;
    }

    const auto fill_case_inputs = [](uint32_t* src_values, uint32_t* dst_values, uint32_t element_count,
                                     uint32_t stride) {
        const uint32_t physical_count = element_count * stride;
        fill_source_values(src_values, physical_count);
        fill_sentinel_values(dst_values, physical_count);
    };

    const auto validate_case_outputs = [](const uint32_t* src_values, const uint32_t* dst_values,
                                          uint32_t element_count, uint32_t stride) {
        const uint32_t physical_count = element_count * stride;
        return validate_case_values(src_values, dst_values, physical_count, stride);
    };

    for (const uint32_t stride : kStrideValues) {
        const uint64_t physical_span_bytes_u64 = static_cast<uint64_t>(logical_count) * stride * sizeof(uint32_t);
        if (physical_span_bytes_u64 > static_cast<uint64_t>(config.max_buffer_bytes)) {
            std::cerr << "[" << kExperimentId << "] Skipping stride " << stride
                      << " because the physical span exceeds the configured buffer budget.\n";
            continue;
        }

        run_stride_case(context, runner, pipeline_resources, logical_count, stride,
                        static_cast<VkDeviceSize>(physical_span_bytes_u64), verbose_progress, fill_case_inputs,
                        validate_case_outputs, output);
    }

    destroy_pipeline_resources(context, pipeline_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
