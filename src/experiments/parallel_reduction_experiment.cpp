#include "experiments/parallel_reduction_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "21_parallel_reduction";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kTreeReductionLevels = 8U;
constexpr VkDeviceSize kInputElementBytes = sizeof(uint32_t);
constexpr VkDeviceSize kOutputBytes = sizeof(uint32_t);
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 31U;
constexpr uint32_t kInputPatternBase = 1U;

enum class ReductionStrategy {
    GlobalAtomic,
    SharedTree,
};

struct VariantDescriptor {
    ReductionStrategy strategy;
    const char* variant_name;
    const char* shader_filename;
    uint32_t staged_reduction_levels;
    uint32_t shared_bytes_per_workgroup;
};

constexpr std::array<VariantDescriptor, 2> kVariantDescriptors = {{
    {ReductionStrategy::GlobalAtomic, "global_atomic", "21_parallel_reduction_global_atomic.comp.spv", 0U, 0U},
    {ReductionStrategy::SharedTree, "shared_tree", "21_parallel_reduction_shared_tree.comp.spv", kTreeReductionLevels,
     kWorkgroupSize* static_cast<uint32_t>(sizeof(uint32_t))},
}};

struct BufferResources {
    BufferResource input_device{};
    BufferResource output_device{};
    BufferResource staging{};
    void* staging_mapped_ptr = nullptr;
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
    uint32_t element_count = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    return (((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus) + kInputPatternBase;
}

void fill_input_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = input_pattern_value(index);
    }
}

uint32_t compute_reference_sum(const uint32_t* values, uint32_t element_count) {
    uint64_t total = 0U;
    for (uint32_t index = 0U; index < element_count; ++index) {
        total += values[index];
    }
    return static_cast<uint32_t>(total);
}

bool validate_output_sum(uint32_t gpu_value, uint32_t reference_value) {
    return gpu_value == reference_value;
}

std::string make_variant_name(ReductionStrategy strategy) {
    switch (strategy) {
    case ReductionStrategy::GlobalAtomic:
        return "global_atomic";
    case ReductionStrategy::SharedTree:
        return "shared_tree";
    }

    return "unknown";
}

std::string make_case_name(const std::string& variant_name, uint32_t problem_size) {
    return std::string(kExperimentId) + "_" + variant_name + "_elements_" + std::to_string(problem_size);
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.staging_mapped_ptr != nullptr && resources.staging.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging.memory);
        resources.staging_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.staging);
    destroy_buffer_resource(context.device(), resources.output_device);
    destroy_buffer_resource(context.device(), resources.input_device);
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize input_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), input_bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.input_device)) {
        std::cerr << "Failed to create parallel reduction input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), kOutputBytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.output_device)) {
        std::cerr << "Failed to create parallel reduction output buffer.\n";
        destroy_buffer_resources(context, out_resources);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), input_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.staging)) {
        std::cerr << "Failed to create parallel reduction staging buffer.\n";
        destroy_buffer_resources(context, out_resources);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.staging, "parallel reduction staging buffer",
                           out_resources.staging_mapped_ptr)) {
        destroy_buffer_resources(context, out_resources);
        return false;
    }

    return true;
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_device.buffer, 0U, buffers.input_device.size};
    const VkDescriptorBufferInfo output_info{buffers.output_device.buffer, 0U, buffers.output_device.size};

    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = input_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 1U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = output_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load parallel reduction shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create parallel reduction descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create parallel reduction descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate parallel reduction descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create parallel reduction pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create parallel reduction compute pipeline.\n";
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

std::vector<uint32_t> make_problem_sizes(uint64_t max_elements) {
    constexpr std::array<std::pair<uint64_t, uint64_t>, 6> kFractions = {{
        {1U, 16U},
        {1U, 8U},
        {1U, 4U},
        {1U, 2U},
        {3U, 4U},
        {1U, 1U},
    }};

    std::vector<uint32_t> sizes;
    sizes.reserve(kFractions.size());

    for (const auto [numerator, denominator] : kFractions) {
        if (max_elements == 0U || denominator == 0U) {
            continue;
        }

        uint64_t candidate = (max_elements * numerator) / denominator;
        if (candidate == 0U) {
            candidate = 1U;
        }

        candidate = std::min(candidate, max_elements);
        sizes.push_back(static_cast<uint32_t>(candidate));
    }

    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

VkDeviceSize compute_input_bytes(uint32_t problem_size) {
    return static_cast<VkDeviceSize>(problem_size) * kInputElementBytes;
}

VkDeviceSize compute_logical_bytes_touched(uint32_t problem_size) {
    return compute_input_bytes(problem_size) + kOutputBytes;
}

double run_upload_stage(VulkanContext& context, const BufferResource& staging_buffer,
                        const BufferResource& input_buffer, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, bytes};
        vkCmdCopyBuffer(command_buffer, staging_buffer.buffer, input_buffer.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, input_buffer.buffer,
                                                                                bytes);
    });
}

double run_dispatch_stage(VulkanContext& context, const PipelineResources& resources,
                          const BufferResource& output_buffer, uint32_t element_count, uint32_t group_count_x) {
    const PushConstants push_constants{element_count};

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdFillBuffer(command_buffer, output_buffer.buffer, 0U, kOutputBytes, 0U);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, output_buffer.buffer,
                                                                                kOutputBytes);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, output_buffer.buffer,
                                                                          kOutputBytes);
    });
}

double run_readback_stage(VulkanContext& context, const BufferResource& output_buffer,
                          const BufferResource& staging_buffer) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{0U, 0U, kOutputBytes};
        vkCmdCopyBuffer(command_buffer, output_buffer.buffer, staging_buffer.buffer, 1U, &copy_region);
    });
}

void record_case_notes(std::string& notes, ReductionStrategy strategy, uint32_t problem_size, uint32_t group_count_x,
                       bool rounded_from_max, bool correctness_pass, bool dispatch_ok) {
    const VkDeviceSize input_bytes = compute_input_bytes(problem_size);
    const VkDeviceSize logical_bytes_touched = compute_logical_bytes_touched(problem_size);
    const VkDeviceSize estimated_atomic_bytes = (strategy == ReductionStrategy::GlobalAtomic)
                                                    ? static_cast<VkDeviceSize>(problem_size) * kOutputBytes
                                                    : static_cast<VkDeviceSize>(group_count_x) * kOutputBytes;

    append_note(notes, std::string("reduction_strategy=") + make_variant_name(strategy));
    append_note(notes, "workgroup_size=" + std::to_string(kWorkgroupSize));
    append_note(notes, "staged_reduction_levels=" +
                           std::to_string(strategy == ReductionStrategy::SharedTree ? kTreeReductionLevels : 0U));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "input_elements=" + std::to_string(problem_size));
    append_note(notes, "input_bytes=" + std::to_string(static_cast<unsigned long long>(input_bytes)));
    append_note(notes, "output_bytes=" + std::to_string(static_cast<unsigned long long>(kOutputBytes)));
    append_note(notes,
                "logical_bytes_touched=" + std::to_string(static_cast<unsigned long long>(logical_bytes_touched)));
    append_note(notes,
                "estimated_atomic_bytes=" + std::to_string(static_cast<unsigned long long>(estimated_atomic_bytes)));
    append_note(notes, "shared_bytes_per_workgroup=" +
                           std::to_string(strategy == ReductionStrategy::SharedTree
                                              ? static_cast<unsigned long long>(kWorkgroupSize * sizeof(uint32_t))
                                              : 0ULL));
    append_note(notes, "output_reset_included_in_dispatch=true");
    if (rounded_from_max) {
        append_note(notes, "problem_size_rounded_from_capacity=true");
    }
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, ReductionStrategy strategy, uint32_t problem_size,
              bool rounded_from_max, ParallelReductionExperimentOutput& output, bool verbose_progress) {
    const std::string variant_name = make_variant_name(strategy);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(problem_size, kWorkgroupSize);
    auto* staging_values = static_cast<uint32_t*>(buffers.staging_mapped_ptr);
    if (staging_values == nullptr || group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Missing mapped staging pointer or illegal dispatch size.\n";
        return false;
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", input_elements=" << problem_size << ", group_count_x=" << group_count_x
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const VkDeviceSize input_bytes = compute_input_bytes(problem_size);

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_input_values(staging_values, problem_size);
        const uint32_t reference_sum = compute_reference_sum(staging_values, problem_size);
        const double upload_ms = run_upload_stage(context, buffers.staging, buffers.input_device, input_bytes);
        const double dispatch_ms =
            run_dispatch_stage(context, pipeline_resources, buffers.output_device, problem_size, group_count_x);
        const double readback_ms = run_readback_stage(context, buffers.output_device, buffers.staging);
        const bool correctness_pass = validate_output_sum(staging_values[0], reference_sum);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                      << ", readback_ms=" << readback_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }

        if (!std::isfinite(upload_ms) || !std::isfinite(dispatch_ms) || !std::isfinite(readback_ms) ||
            !correctness_pass) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", input_elements=" << problem_size
                      << ", dispatch_ok=" << (std::isfinite(dispatch_ms) ? "true" : "false")
                      << ", correctness=" << (correctness_pass ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        fill_input_values(staging_values, problem_size);
        const uint32_t reference_sum = compute_reference_sum(staging_values, problem_size);

        const auto start = std::chrono::high_resolution_clock::now();
        const double upload_ms = run_upload_stage(context, buffers.staging, buffers.input_device, input_bytes);
        const double dispatch_ms =
            run_dispatch_stage(context, pipeline_resources, buffers.output_device, problem_size, group_count_x);
        const double readback_ms = run_readback_stage(context, buffers.output_device, buffers.staging);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool upload_ok = std::isfinite(upload_ms);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool readback_ok = std::isfinite(readback_ms);
        const bool correctness_pass =
            upload_ok && dispatch_ok && readback_ok && validate_output_sum(staging_values[0], reference_sum);

        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, strategy, problem_size, group_count_x, rounded_from_max, correctness_pass,
                          dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", upload_ms=" << upload_ms << ", dispatch_ms=" << dispatch_ms
                      << ", readback_ms=" << readback_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = problem_size,
            .dispatch_count = 1U,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(problem_size, 1U, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(compute_logical_bytes_touched(problem_size), dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(make_case_name(variant_name, problem_size), dispatch_samples);
    output.summary_results.push_back(summary);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", input_elements=" << problem_size << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

ParallelReductionExperimentOutput run_parallel_reduction_experiment(VulkanContext& context,
                                                                    const BenchmarkRunner& runner,
                                                                    const ParallelReductionExperimentConfig& config) {
    ParallelReductionExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "parallel reduction experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        const auto& descriptor = kVariantDescriptors[index];
        const std::string user_path = descriptor.strategy == ReductionStrategy::GlobalAtomic
                                          ? config.global_atomic_shader_path
                                          : config.shared_tree_shader_path;
        shader_paths[index] = VulkanComputeUtils::resolve_shader_path(user_path, descriptor.shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for parallel reduction variant " << descriptor.variant_name
                      << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint64_t usable_bytes =
        config.max_buffer_bytes > kOutputBytes ? (config.max_buffer_bytes - kOutputBytes) : 0U;
    const uint64_t buffer_limited_elements = usable_bytes / (kInputElementBytes * 2U);
    const uint64_t dispatch_limited_elements =
        static_cast<uint64_t>(device_properties.limits.maxComputeWorkGroupCount[0]) * kWorkgroupSize;
    const uint64_t max_elements = std::min(buffer_limited_elements, dispatch_limited_elements);
    if (max_elements == 0U) {
        std::cerr << "Scratch buffer too small for parallel reduction experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint64_t max_supported_elements =
        std::min<uint64_t>(max_elements, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> problem_sizes = make_problem_sizes(max_supported_elements);
    if (problem_sizes.empty()) {
        std::cerr << "No legal problem sizes available for parallel reduction experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Global atomic shader: " << shader_paths[0] << "\n";
        std::cout << "[" << kExperimentId << "] Shared tree shader: " << shader_paths[1] << "\n";
        std::cout << "[" << kExperimentId << "] input_capacity_elements=" << max_elements
                  << ", buffer_limited_elements=" << buffer_limited_elements
                  << ", dispatch_limited_elements=" << dispatch_limited_elements
                  << ", scratch_size_bytes=" << config.max_buffer_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const VkDeviceSize input_buffer_bytes = static_cast<VkDeviceSize>(max_supported_elements) * kInputElementBytes;
    BufferResources buffers{};
    if (!create_buffer_resources(context, input_buffer_bytes, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* staging_values = static_cast<uint32_t*>(buffers.staging_mapped_ptr);
    if (staging_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped staging pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    std::array<PipelineResources, kVariantDescriptors.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (const uint32_t problem_size : problem_sizes) {
        for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
            const auto& descriptor = kVariantDescriptors[index];
            const bool rounded_from_max = static_cast<uint64_t>(problem_size) != max_supported_elements;
            if (!run_case(context, runner, buffers, pipeline_resources[index], descriptor.strategy, problem_size,
                          rounded_from_max, output, config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
