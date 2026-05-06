#include "experiments/scalar_type_width_sweep_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/scalar_type_width_utils.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

constexpr const char* kExperimentId = "10_scalar_type_width_sweep";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;

using ScalarTypeWidthUtils::buffer_size_for_variant;
using ScalarTypeWidthUtils::dequantize_u16;
using ScalarTypeWidthUtils::dequantize_u8;
using ScalarTypeWidthUtils::expected_variant_value;
using ScalarTypeWidthUtils::float_to_half_bits;
using ScalarTypeWidthUtils::half_bits_to_float;
using ScalarTypeWidthUtils::make_seed_scalar;
using ScalarTypeWidthUtils::quantize_u16;
using ScalarTypeWidthUtils::quantize_u8;
using ScalarTypeWidthUtils::read_u16_lane;
using ScalarTypeWidthUtils::read_u8_lane;
using ScalarTypeWidthUtils::storage_bytes_per_element;
using ScalarTypeWidthUtils::storage_units_for_variant;
using ScalarTypeWidthUtils::validation_tolerance;
using ScalarTypeWidthUtils::WidthVariant;
using ScalarTypeWidthUtils::write_u16_lane;
using ScalarTypeWidthUtils::write_u8_lane;

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

struct ValidationStats {
    bool pass = true;
    float max_abs_error = 0.0F;
    double mean_abs_error = 0.0;
};

std::vector<uint32_t> make_element_counts(uint32_t max_elements) {
    const std::array<uint32_t, 8> base_counts = {1310720U,  2621440U,  5242880U,  10485760U,
                                                 20971520U, 41943040U, 83886080U, 167772160U};
    std::vector<uint32_t> output;
    output.reserve(base_counts.size() + 5U);

    for (const uint32_t value : base_counts) {
        if (value <= max_elements) {
            output.push_back(value);
        }
    }

    if (output.empty()) {
        const std::array<uint32_t, 4> fallback_counts = {4096U, 8192U, 16384U, 32768U};
        for (const uint32_t value : fallback_counts) {
            if (value <= max_elements) {
                output.push_back(value);
            }
        }
    }

    if (max_elements > 0U && (output.empty() || output.back() != max_elements)) {
        output.push_back(max_elements);
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

template <typename ReadActualFn>
ValidationStats validate_variant_values(WidthVariant variant, uint32_t count, ReadActualFn read_actual) {
    ValidationStats stats{};
    const float tolerance = validation_tolerance(variant);
    double error_sum = 0.0;

    for (uint32_t index = 0U; index < count; ++index) {
        const float actual = read_actual(index);
        if (!std::isfinite(actual)) {
            stats.pass = false;
            stats.max_abs_error = std::numeric_limits<float>::infinity();
            continue;
        }

        const float expected = expected_variant_value(variant, index);
        const float abs_error = std::fabs(actual - expected);
        stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
        error_sum += static_cast<double>(abs_error);

        if (abs_error > tolerance) {
            stats.pass = false;
        }
    }

    if (count > 0U) {
        stats.mean_abs_error = error_sum / static_cast<double>(count);
    }

    return stats;
}

void fill_fp32_seed_values(float* values, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        values[index] = make_seed_scalar(index);
    }
}

void fill_u32_seed_values(uint32_t* values, uint32_t count) {
    for (uint32_t index = 0U; index < count; ++index) {
        values[index] = std::bit_cast<uint32_t>(make_seed_scalar(index));
    }
}

void fill_fp16_seed_values(uint32_t* words, uint32_t count) {
    const uint32_t word_count = storage_units_for_variant(WidthVariant::kFp16Storage, count);
    std::fill(words, words + word_count, 0U);

    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t word_index = index / 2U;
        const uint32_t lane = index % 2U;
        write_u16_lane(words[word_index], lane, float_to_half_bits(make_seed_scalar(index)));
    }
}

void fill_u16_seed_values(uint32_t* words, uint32_t count) {
    const uint32_t word_count = storage_units_for_variant(WidthVariant::kU16, count);
    std::fill(words, words + word_count, 0U);

    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t word_index = index / 2U;
        const uint32_t lane = index % 2U;
        write_u16_lane(words[word_index], lane, quantize_u16(make_seed_scalar(index)));
    }
}

void fill_u8_seed_values(uint32_t* words, uint32_t count) {
    const uint32_t word_count = storage_units_for_variant(WidthVariant::kU8, count);
    std::fill(words, words + word_count, 0U);

    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t word_index = index / 4U;
        const uint32_t lane = index % 4U;
        write_u8_lane(words[word_index], lane, quantize_u8(make_seed_scalar(index)));
    }
}

ValidationStats validate_fp32_values(const float* values, uint32_t count) {
    return validate_variant_values(WidthVariant::kFp32, count, [&](uint32_t index) {
        return values[index];
    });
}

ValidationStats validate_u32_values(const uint32_t* values, uint32_t count) {
    return validate_variant_values(WidthVariant::kU32, count, [&](uint32_t index) {
        return std::bit_cast<float>(values[index]);
    });
}

ValidationStats validate_fp16_values(const uint32_t* words, uint32_t count) {
    return validate_variant_values(WidthVariant::kFp16Storage, count, [&](uint32_t index) {
        const uint32_t word_index = index / 2U;
        const uint32_t lane = index % 2U;
        const uint16_t bits = read_u16_lane(words[word_index], lane);
        return half_bits_to_float(bits);
    });
}

ValidationStats validate_u16_values(const uint32_t* words, uint32_t count) {
    return validate_variant_values(WidthVariant::kU16, count, [&](uint32_t index) {
        const uint32_t word_index = index / 2U;
        const uint32_t lane = index % 2U;
        return dequantize_u16(read_u16_lane(words[word_index], lane));
    });
}

ValidationStats validate_u8_values(const uint32_t* words, uint32_t count) {
    return validate_variant_values(WidthVariant::kU8, count, [&](uint32_t index) {
        const uint32_t word_index = index / 4U;
        const uint32_t lane = index % 4U;
        return dequantize_u8(read_u8_lane(words[word_index], lane));
    });
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

double run_dispatch(VulkanContext& context, uint32_t dispatch_elements, uint32_t logical_elements,
                    const SingleBufferPipelineResources& resources) {
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(dispatch_elements, kWorkgroupSize);
    if (group_count == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           sizeof(logical_elements), &logical_elements);
        for (uint32_t dispatch_index = 0U; dispatch_index < kDispatchCount; ++dispatch_index) {
            vkCmdDispatch(command_buffer, group_count, 1U, 1U);
        }
    });
}

double compute_effective_gbps(WidthVariant variant, uint32_t elements, uint32_t dispatch_count,
                              double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double bytes_per_dispatch =
        storage_bytes_per_element(variant) * static_cast<double>(elements) * static_cast<double>(dispatch_count) * 2.0;
    return bytes_per_dispatch / (dispatch_gpu_ms * 1.0e6);
}

std::string build_case_name(const std::string& variant, uint32_t elements) {
    return std::string(kExperimentId) + "_" + variant + "_elements_" + std::to_string(elements);
}

template <typename PrepareFn, typename ValidateFn>
void run_variant_case(VulkanContext& context, const BenchmarkRunner& runner, WidthVariant width_variant,
                      const std::string& variant_name, uint32_t element_count,
                      const SingleBufferPipelineResources& resources, PrepareFn prepare_inputs,
                      ValidateFn validate_outputs, bool verbose_progress,
                      ScalarTypeWidthSweepExperimentOutput& output) {
    const uint32_t dispatch_elements = storage_units_for_variant(width_variant, element_count);
    const uint32_t group_count = VulkanComputeUtils::compute_group_count_1d(dispatch_elements, kWorkgroupSize);
    const float tolerance = validation_tolerance(width_variant);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant start: name=" << variant_name << ", elements=" << element_count
                  << ", dispatch_elements=" << dispatch_elements << ", group_count_x=" << group_count
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        prepare_inputs(element_count);
        const double dispatch_ms = run_dispatch(context, dispatch_elements, element_count, resources);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const ValidationStats stats = dispatch_ok ? validate_outputs(element_count) : ValidationStats{.pass = false};
        const bool data_ok = stats.pass;

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", elements=" << element_count << ", gpu_ms=" << dispatch_ms
                      << ", max_abs_error=" << stats.max_abs_error << ", correctness=" << (data_ok ? "pass" : "fail")
                      << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] warmup issue for variant=" << variant_name
                      << ", elements=" << element_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << ", max_abs_error=" << stats.max_abs_error
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        prepare_inputs(element_count);
        const double dispatch_ms = run_dispatch(context, dispatch_elements, element_count, resources);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const ValidationStats stats = dispatch_ok ? validate_outputs(element_count) : ValidationStats{.pass = false};
        const bool data_ok = stats.pass;

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        std::string notes;
        append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
        append_note(notes, "group_count_x=" + std::to_string(group_count));
        append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
        append_note(notes, "dispatch_elements=" + std::to_string(dispatch_elements));
        append_note(notes, "storage_bytes_per_element=" + std::to_string(storage_bytes_per_element(width_variant)));
        append_note(notes, "fp32_bytes_per_element=4");
        append_note(notes, "storage_ratio_vs_fp32=" + std::to_string(storage_bytes_per_element(width_variant) / 4.0));
        append_note(notes, "validation_tolerance=" + std::to_string(tolerance));
        append_note(notes, "max_abs_error=" + std::to_string(stats.max_abs_error));
        append_note(notes, "mean_abs_error=" + std::to_string(stats.mean_abs_error));
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
                      << " variant=" << variant_name << ", elements=" << element_count << ", gpu_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count() << ", max_abs_error=" << stats.max_abs_error
                      << ", mean_abs_error=" << stats.mean_abs_error
                      << ", correctness=" << (correctness ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = element_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(width_variant, element_count, kDispatchCount, dispatch_ms),
            .correctness_pass = correctness,
            .notes = notes,
        });
    }

    const BenchmarkResult summary =
        BenchmarkRunner::summarize_samples(build_case_name(variant_name, element_count), dispatch_samples);
    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Variant complete: name=" << variant_name
                  << ", elements=" << element_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }
    output.summary_results.push_back(summary);
}

} // namespace

ScalarTypeWidthSweepExperimentOutput
run_scalar_type_width_sweep_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                       const ScalarTypeWidthSweepExperimentConfig& config) {
    ScalarTypeWidthSweepExperimentOutput output{};
    const bool verbose_progress = config.verbose_progress;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "scalar type width sweep experiment requires GPU timestamp support.\n";
        return output;
    }

    const std::string fp32_shader =
        VulkanComputeUtils::resolve_shader_path(config.fp32_shader_path, "10_fp32.comp.spv");
    const std::string fp16_shader =
        VulkanComputeUtils::resolve_shader_path(config.fp16_storage_shader_path, "10_fp16_storage.comp.spv");
    const std::string u32_shader = VulkanComputeUtils::resolve_shader_path(config.u32_shader_path, "10_u32.comp.spv");
    const std::string u16_shader = VulkanComputeUtils::resolve_shader_path(config.u16_shader_path, "10_u16.comp.spv");
    const std::string u8_shader = VulkanComputeUtils::resolve_shader_path(config.u8_shader_path, "10_u8.comp.spv");
    if (fp32_shader.empty() || fp16_shader.empty() || u32_shader.empty() || u16_shader.empty() || u8_shader.empty()) {
        std::cerr << "Could not locate SPIR-V shader(s) for scalar type width sweep experiment.\n";
        return output;
    }

    const uint64_t max_elements_u64 = config.max_buffer_bytes / static_cast<uint64_t>(sizeof(uint32_t));
    if (max_elements_u64 == 0U) {
        std::cerr << "Scratch buffer too small for scalar type width sweep experiment.\n";
        return output;
    }

    const uint32_t max_elements =
        static_cast<uint32_t>(std::min<uint64_t>(max_elements_u64, std::numeric_limits<uint32_t>::max()));
    const std::vector<uint32_t> element_counts = make_element_counts(max_elements);
    if (element_counts.empty()) {
        std::cerr << "No legal element counts available for scalar type width sweep experiment.\n";
        return output;
    }

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Shader fp32: " << fp32_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader fp16_storage: " << fp16_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader u32: " << u32_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader u16: " << u16_shader << "\n";
        std::cout << "[" << kExperimentId << "] Shader u8: " << u8_shader << "\n";
        std::cout << "[" << kExperimentId
                  << "] Storage bytes per element: fp32=4, fp16_storage=2, u32=4, u16=2, u8=1\n";
        std::cout << "[" << kExperimentId << "] Starting run with sizes=" << element_counts.size()
                  << ", variants=5, local_size_x=" << kWorkgroupSize
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    const uint32_t max_count = element_counts.back();

    SingleBufferPipelineResources fp32_resources{};
    if (!create_single_buffer_pipeline_resources(context, fp32_shader,
                                                 buffer_size_for_variant(WidthVariant::kFp32, max_count), "fp32 buffer",
                                                 fp32_resources)) {
        cleanup_single_buffer_resources(context, fp32_resources);
        return output;
    }

    SingleBufferPipelineResources fp16_resources{};
    if (!create_single_buffer_pipeline_resources(context, fp16_shader,
                                                 buffer_size_for_variant(WidthVariant::kFp16Storage, max_count),
                                                 "fp16_storage buffer", fp16_resources)) {
        cleanup_single_buffer_resources(context, fp32_resources);
        cleanup_single_buffer_resources(context, fp16_resources);
        return output;
    }

    SingleBufferPipelineResources u32_resources{};
    if (!create_single_buffer_pipeline_resources(
            context, u32_shader, buffer_size_for_variant(WidthVariant::kU32, max_count), "u32 buffer", u32_resources)) {
        cleanup_single_buffer_resources(context, fp32_resources);
        cleanup_single_buffer_resources(context, fp16_resources);
        cleanup_single_buffer_resources(context, u32_resources);
        return output;
    }

    SingleBufferPipelineResources u16_resources{};
    if (!create_single_buffer_pipeline_resources(
            context, u16_shader, buffer_size_for_variant(WidthVariant::kU16, max_count), "u16 buffer", u16_resources)) {
        cleanup_single_buffer_resources(context, fp32_resources);
        cleanup_single_buffer_resources(context, fp16_resources);
        cleanup_single_buffer_resources(context, u32_resources);
        cleanup_single_buffer_resources(context, u16_resources);
        return output;
    }

    SingleBufferPipelineResources u8_resources{};
    if (!create_single_buffer_pipeline_resources(
            context, u8_shader, buffer_size_for_variant(WidthVariant::kU8, max_count), "u8 buffer", u8_resources)) {
        cleanup_single_buffer_resources(context, fp32_resources);
        cleanup_single_buffer_resources(context, fp16_resources);
        cleanup_single_buffer_resources(context, u32_resources);
        cleanup_single_buffer_resources(context, u16_resources);
        cleanup_single_buffer_resources(context, u8_resources);
        return output;
    }

    auto* fp32_values = static_cast<float*>(fp32_resources.mapped_ptr);
    auto* fp16_words = static_cast<uint32_t*>(fp16_resources.mapped_ptr);
    auto* u32_words = static_cast<uint32_t*>(u32_resources.mapped_ptr);
    auto* u16_words = static_cast<uint32_t*>(u16_resources.mapped_ptr);
    auto* u8_words = static_cast<uint32_t*>(u8_resources.mapped_ptr);
    if (fp32_values == nullptr || fp16_words == nullptr || u32_words == nullptr || u16_words == nullptr ||
        u8_words == nullptr) {
        std::cerr << "Mapped pointers are missing for scalar type width sweep experiment resources.\n";
        cleanup_single_buffer_resources(context, fp32_resources);
        cleanup_single_buffer_resources(context, fp16_resources);
        cleanup_single_buffer_resources(context, u32_resources);
        cleanup_single_buffer_resources(context, u16_resources);
        cleanup_single_buffer_resources(context, u8_resources);
        return output;
    }

    for (const uint32_t element_count : element_counts) {
        run_variant_case(
            context, runner, WidthVariant::kFp32, "fp32", element_count, fp32_resources,
            [&](uint32_t count) {
                fill_fp32_seed_values(fp32_values, count);
            },
            [&](uint32_t count) {
                return validate_fp32_values(fp32_values, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, WidthVariant::kFp16Storage, "fp16_storage", element_count, fp16_resources,
            [&](uint32_t count) {
                fill_fp16_seed_values(fp16_words, count);
            },
            [&](uint32_t count) {
                return validate_fp16_values(fp16_words, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, WidthVariant::kU32, "u32", element_count, u32_resources,
            [&](uint32_t count) {
                fill_u32_seed_values(u32_words, count);
            },
            [&](uint32_t count) {
                return validate_u32_values(u32_words, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, WidthVariant::kU16, "u16", element_count, u16_resources,
            [&](uint32_t count) {
                fill_u16_seed_values(u16_words, count);
            },
            [&](uint32_t count) {
                return validate_u16_values(u16_words, count);
            },
            verbose_progress, output);

        run_variant_case(
            context, runner, WidthVariant::kU8, "u8", element_count, u8_resources,
            [&](uint32_t count) {
                fill_u8_seed_values(u8_words, count);
            },
            [&](uint32_t count) {
                return validate_u8_values(u8_words, count);
            },
            verbose_progress, output);
    }

    cleanup_single_buffer_resources(context, fp32_resources);
    cleanup_single_buffer_resources(context, fp16_resources);
    cleanup_single_buffer_resources(context, u32_resources);
    cleanup_single_buffer_resources(context, u16_resources);
    cleanup_single_buffer_resources(context, u8_resources);

    if (verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
