#include "experiments/two_dimensional_locality_transpose_study_experiment.hpp"

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

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "33_two_dimensional_locality_transpose_study";
constexpr uint32_t kTileSize = 16U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kInputPatternMultiplier = 17U;
constexpr uint32_t kInputPatternOffset = 23U;
constexpr uint32_t kInputPatternModulus = 251U;
constexpr uint32_t kOutputSentinelValue = 0xA5A5A5A5U;

enum class VariantKind : uint32_t {
    RowMajorCopy,
    NaiveTranspose,
    TiledTranspose,
    TiledTransposePadded,
};

struct VariantDescriptor {
    VariantKind kind;
    const char* variant_name;
    const char* shader_filename;
};

constexpr std::array<VariantDescriptor, 4> kVariantDescriptors = {{
    {VariantKind::RowMajorCopy, "row_major_copy", "33_row_major_copy.comp.spv"},
    {VariantKind::NaiveTranspose, "naive_transpose", "33_naive_transpose.comp.spv"},
    {VariantKind::TiledTranspose, "tiled_transpose", "33_tiled_transpose.comp.spv"},
    {VariantKind::TiledTransposePadded, "tiled_transpose_padded", "33_tiled_transpose_padded.comp.spv"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource output_buffer{};
    void* input_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
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
    uint32_t matrix_dim = 0U;
};

static_assert(sizeof(PushConstants) == sizeof(uint32_t));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t input_pattern_value(uint32_t index) {
    return ((index * kInputPatternMultiplier) + kInputPatternOffset) % kInputPatternModulus;
}

uint32_t transform_value(uint32_t input_value, uint32_t src_x, uint32_t src_y) {
    return (input_value ^ (src_x * 131U)) + (src_y * 17U) + 0x9E3779B9U;
}

void fill_input_values(uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        values[index] = input_pattern_value(index);
    }
}

void fill_output_values(uint32_t* values, uint32_t element_count) {
    std::fill_n(values, element_count, kOutputSentinelValue);
}

bool validate_input_values(const uint32_t* values, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        if (values[index] != input_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

uint32_t determine_matrix_dim(std::size_t max_buffer_bytes) {
    const uint64_t max_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    if (max_elements < static_cast<uint64_t>(kTileSize) * kTileSize) {
        return 0U;
    }

    const uint64_t raw_dim = static_cast<uint64_t>(std::sqrt(static_cast<double>(max_elements)));
    const uint64_t rounded_dim = raw_dim - (raw_dim % kTileSize);
    return rounded_dim >= kTileSize ? static_cast<uint32_t>(rounded_dim) : 0U;
}

VkDeviceSize compute_span_bytes(uint32_t matrix_dim) {
    const uint64_t element_count = static_cast<uint64_t>(matrix_dim) * matrix_dim;
    return static_cast<VkDeviceSize>(element_count * sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create 2D locality input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create 2D locality output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "2D locality input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "2D locality output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, input_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load 2D locality shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create 2D locality descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create 2D locality descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate 2D locality descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create 2D locality pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create 2D locality compute pipeline.\n";
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

std::vector<uint32_t> build_reference_values(const uint32_t* input_values, uint32_t matrix_dim, VariantKind kind) {
    const uint32_t element_count = matrix_dim * matrix_dim;
    std::vector<uint32_t> reference(element_count, kOutputSentinelValue);

    for (uint32_t y = 0U; y < matrix_dim; ++y) {
        for (uint32_t x = 0U; x < matrix_dim; ++x) {
            const uint32_t src_index = (y * matrix_dim) + x;
            const uint32_t transformed = transform_value(input_values[src_index], x, y);
            if (kind == VariantKind::RowMajorCopy) {
                reference[src_index] = transformed;
            } else {
                const uint32_t dst_index = (x * matrix_dim) + y;
                reference[dst_index] = transformed;
            }
        }
    }

    return reference;
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (output_values[index] != reference_values[index]) {
            return false;
        }
    }
    return true;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, uint32_t matrix_dim) {
    if (matrix_dim == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const uint32_t group_count_x = (matrix_dim + kTileSize - 1U) / kTileSize;
    const PushConstants push_constants{matrix_dim};
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_constants)), &push_constants);
        vkCmdDispatch(command_buffer, group_count_x, group_count_x, 1U);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t matrix_dim,
                       bool correctness_pass, bool dispatch_ok, std::size_t scratch_size_bytes) {
    append_note(notes, std::string("variant_kind=") + descriptor.variant_name);
    append_note(notes, "matrix_dim=" + std::to_string(matrix_dim));
    append_note(notes, "tile_size=" + std::to_string(kTileSize));
    append_note(notes, "transpose=" + std::string(descriptor.kind == VariantKind::RowMajorCopy ? "false" : "true"));
    append_note(notes, "payload_bytes_per_element=8");
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    if (descriptor.kind == VariantKind::TiledTranspose) {
        append_note(notes, "shared_tile_bytes=" + std::to_string(static_cast<unsigned long long>(kTileSize * kTileSize *
                                                                                                 sizeof(uint32_t))));
    }
    if (descriptor.kind == VariantKind::TiledTransposePadded) {
        append_note(notes, "shared_tile_bytes=" + std::to_string(static_cast<unsigned long long>(
                                                      kTileSize * (kTileSize + 1U) * sizeof(uint32_t))));
    }
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, const VariantDescriptor& descriptor, uint32_t matrix_dim,
              TwoDimensionalLocalityTransposeStudyExperimentOutput& output, bool verbose_progress,
              std::size_t scratch_size_bytes) {
    const uint32_t element_count = matrix_dim * matrix_dim;
    const auto* input_values = static_cast<const uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped input or output buffer pointer for variant=" << descriptor.variant_name << ".\n";
        return false;
    }

    const std::vector<uint32_t> reference_values = build_reference_values(input_values, matrix_dim, descriptor.kind);
    const uint64_t payload_bytes = static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 2U;
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_output_values(output_values, element_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, matrix_dim);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, element_count) &&
                             validate_output_values(output_values, reference_values);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", matrix_dim=" << matrix_dim
                      << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        fill_output_values(output_values, element_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, matrix_dim);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass = dispatch_ok && validate_input_values(input_values, element_count) &&
                                      validate_output_values(output_values, reference_values);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, matrix_dim, correctness_pass, dispatch_ok, scratch_size_bytes);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_dim_" + std::to_string(matrix_dim),
        dispatch_samples));
    return true;
}

} // namespace

TwoDimensionalLocalityTransposeStudyExperimentOutput run_two_dimensional_locality_transpose_study_experiment(
    VulkanContext& context, const BenchmarkRunner& runner,
    const TwoDimensionalLocalityTransposeStudyExperimentConfig& config) {
    TwoDimensionalLocalityTransposeStudyExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "2D locality transpose study requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kVariantDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        const auto& descriptor = kVariantDescriptors[index];
        std::string user_path;
        switch (descriptor.kind) {
        case VariantKind::RowMajorCopy:
            user_path = config.row_major_copy_shader_path;
            break;
        case VariantKind::NaiveTranspose:
            user_path = config.naive_transpose_shader_path;
            break;
        case VariantKind::TiledTranspose:
            user_path = config.tiled_transpose_shader_path;
            break;
        case VariantKind::TiledTransposePadded:
            user_path = config.tiled_transpose_padded_shader_path;
            break;
        }
        shader_paths[index] = VulkanComputeUtils::resolve_shader_path(user_path, descriptor.shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for 2D locality variant " << descriptor.variant_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    const uint32_t matrix_dim = determine_matrix_dim(config.max_buffer_bytes);
    if (matrix_dim == 0U) {
        std::cerr << "Scratch buffer too small for 2D locality transpose study.\n";
        output.all_points_correct = false;
        return output;
    }

    const uint32_t element_count = matrix_dim * matrix_dim;
    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_span_bytes(matrix_dim), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (input_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped input or output buffer pointer.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    fill_input_values(input_values, element_count);
    fill_output_values(output_values, element_count);

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

    for (std::size_t index = 0; index < kVariantDescriptors.size(); ++index) {
        if (!run_case(context, runner, buffers, pipeline_resources[index], kVariantDescriptors[index], matrix_dim,
                      output, config.verbose_progress, config.scratch_size_bytes)) {
            output.all_points_correct = false;
        }
    }

    for (PipelineResources& resources : pipeline_resources) {
        destroy_pipeline_resources(context, resources);
    }
    destroy_buffer_resources(context, buffers);
    return output;
}
