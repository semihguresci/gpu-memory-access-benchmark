#include "experiments/frame_to_frame_coherence_experiment.hpp"

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

constexpr const char* kExperimentId = "37_frame_to_frame_coherence";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kFramesPerIteration = 8U;
constexpr uint32_t kTargetElementCount = 67108864U;
constexpr uint32_t kSourceSeed = 0x31415926U;
constexpr uint32_t kModeRandomSeed = 0x9E3779B9U;
constexpr uint32_t kOutputSentinel = 0xDEADBEEFU;
constexpr uint32_t kHashMul0 = 0x7FEB352DU;
constexpr uint32_t kHashMul1 = 0x846CA68BU;
constexpr uint32_t kCoherentFrameShift = 17U;
constexpr uint32_t kBlockLaneCount = 32U;

enum class CoherenceMode : uint32_t {
    CoherentShift = 0U,
    BlockScramble = 1U,
    FrameRandom = 2U,
};

struct VariantDescriptor {
    CoherenceMode mode;
    const char* variant_name;
    const char* locality_profile;
};

constexpr std::array<VariantDescriptor, 3> kVariantDescriptors = {{
    {CoherenceMode::CoherentShift, "coherent_shift", "high"},
    {CoherenceMode::BlockScramble, "block_scramble", "medium"},
    {CoherenceMode::FrameRandom, "frame_random", "low"},
}};

struct BufferResources {
    BufferResource source_buffer{};
    BufferResource output_buffer{};
    void* source_mapped_ptr = nullptr;
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
    uint32_t element_count = 0U;
    uint32_t frame_index = 0U;
    uint32_t coherence_mode = 0U;
    uint32_t random_seed = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint32_t hash32(uint32_t value) {
    value ^= value >> 16U;
    value *= kHashMul0;
    value ^= value >> 15U;
    value *= kHashMul1;
    value ^= value >> 16U;
    return value;
}

uint32_t source_pattern_value(uint32_t index) {
    return hash32(index ^ kSourceSeed);
}

uint32_t resolve_source_index(uint32_t index, uint32_t frame_index, uint32_t element_count, CoherenceMode mode) {
    switch (mode) {
    case CoherenceMode::CoherentShift:
        return (index + (frame_index * kCoherentFrameShift)) % element_count;
    case CoherenceMode::BlockScramble: {
        const uint32_t block_base = (index / kBlockLaneCount) * kBlockLaneCount;
        const uint32_t lane = index % kBlockLaneCount;
        const uint32_t lane_rotated = (lane + frame_index) % kBlockLaneCount;
        uint32_t resolved = block_base + lane_rotated;
        if (resolved >= element_count) {
            resolved = element_count - 1U;
        }
        return resolved;
    }
    case CoherenceMode::FrameRandom:
        return hash32(index ^ (frame_index * kModeRandomSeed) ^ kModeRandomSeed) % element_count;
    }
    return index;
}

uint32_t expected_output_value(const uint32_t* source_values, uint32_t index, uint32_t frame_index,
                               uint32_t element_count, CoherenceMode mode) {
    const uint32_t source_index = resolve_source_index(index, frame_index, element_count, mode);
    const uint32_t input_value = source_values[source_index];
    uint32_t state = input_value ^ (frame_index * 0x85EBCA6BU) ^ (index * 0x9E3779B9U);
    state ^= state >> 16U;
    state *= 0x7FEB352DU;
    state ^= state >> 15U;
    state *= 0x846CA68BU;
    state ^= state >> 16U;
    return state;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t buffer_capacity_elements = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity_elements = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped_elements =
        std::min({buffer_capacity_elements, dispatch_capacity_elements, static_cast<uint64_t>(kTargetElementCount),
                  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    if (capped_elements < kWorkgroupSize) {
        return 0U;
    }
    return static_cast<uint32_t>(capped_elements - (capped_elements % kWorkgroupSize));
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_elements) {
    const std::array<uint32_t, 6> base_sizes = {1048576U, 4194304U, 8388608U, 16777216U, 33554432U, 67108864U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_elements) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_elements) {
        sizes.push_back(max_elements);
    }
    return sizes;
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

bool create_buffer_resources(VulkanContext& context, uint32_t max_elements, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_elements),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.source_buffer)) {
        std::cerr << "Failed to create frame coherence source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_span_bytes(max_elements),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create frame coherence output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.source_buffer, "frame coherence source buffer",
                           out_resources.source_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "frame coherence output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.source_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.source_buffer.memory);
            out_resources.source_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.source_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.source_mapped_ptr != nullptr && resources.source_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.source_buffer.memory);
        resources.source_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.source_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo source_info{buffers.source_buffer.buffer, 0U, buffers.source_buffer.size};
    const VkDescriptorBufferInfo output_info{buffers.output_buffer.buffer, 0U, buffers.output_buffer.size};
    // Shader binding contract:
    //   0 -> stable source dataset reused across frames
    //   1 -> current-frame outputs for each logical invocation
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 0U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = source_info,
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
        std::cerr << "Failed to load frame coherence shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create frame coherence descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create frame coherence descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate frame coherence descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create frame coherence pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create frame coherence compute pipeline.\n";
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

void fill_source_values(uint32_t* source_values, uint32_t max_elements) {
    for (uint32_t index = 0U; index < max_elements; ++index) {
        source_values[index] = source_pattern_value(index);
    }
}

bool validate_source_values(const uint32_t* source_values, uint32_t max_elements) {
    for (uint32_t index = 0U; index < max_elements; ++index) {
        if (source_values[index] != source_pattern_value(index)) {
            return false;
        }
    }
    return true;
}

void record_compute_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U,
                         1U, &barrier, 0U, nullptr, 0U, nullptr);
}

double run_dispatch_sequence(VulkanContext& context, const PipelineResources& pipeline_resources,
                             uint32_t element_count, CoherenceMode mode) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);

        for (uint32_t frame = 0U; frame < kFramesPerIteration; ++frame) {
            // coherence_mode mirrors the shader-side source-index resolver:
            //   0 = coherent shift
            //   1 = block-local lane rotation
            //   2 = per-frame random remap
            const PushConstants constants{
                .element_count = element_count,
                .frame_index = frame,
                .coherence_mode = static_cast<uint32_t>(mode),
                .random_seed = kModeRandomSeed,
            };
            vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                               static_cast<uint32_t>(sizeof(constants)), &constants);
            vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
            if (frame + 1U < kFramesPerIteration) {
                record_compute_barrier(command_buffer);
            }
        }
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& variant, uint32_t element_count,
                       bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "coherence_mode=" + std::string(variant.variant_name));
    append_note(notes, "locality_profile=" + std::string(variant.locality_profile));
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "frames_per_iteration=" + std::to_string(kFramesPerIteration));
    append_note(notes, "coherent_shift=" + std::to_string(kCoherentFrameShift));
    append_note(notes, "block_lane_count=" + std::to_string(kBlockLaneCount));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, const VariantDescriptor& variant, uint32_t element_count,
              FrameToFrameCoherenceExperimentOutput& output, bool verbose_progress) {
    const auto* source_values = static_cast<const uint32_t*>(buffers.source_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    if (source_values == nullptr || output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped buffers for variant=" << variant.variant_name << ".\n";
        return false;
    }

    const uint32_t last_frame = kFramesPerIteration - 1U;
    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t index = 0U; index < element_count; ++index) {
        reference[index] = expected_output_value(source_values, index, last_frame, element_count, variant.mode);
    }

    const uint64_t payload_bytes =
        static_cast<uint64_t>(element_count) * sizeof(uint32_t) * 2ULL * static_cast<uint64_t>(kFramesPerIteration);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));
    const bool source_valid = validate_source_values(source_values, element_count);

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        std::fill_n(output_values, element_count, kOutputSentinel);
        const double dispatch_ms = run_dispatch_sequence(context, pipeline_resources, element_count, variant.mode);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            bool correctness_pass = dispatch_ok && source_valid;
            if (correctness_pass) {
                correctness_pass = std::equal(output_values, output_values + element_count, reference.begin());
            }
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << variant.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        std::fill_n(output_values, element_count, kOutputSentinel);
        const double dispatch_ms = run_dispatch_sequence(context, pipeline_resources, element_count, variant.mode);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && source_valid && std::equal(output_values, output_values + element_count, reference.begin());
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, variant, element_count, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << variant.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant.variant_name,
            .problem_size = element_count,
            .dispatch_count = kFramesPerIteration,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, kFramesPerIteration, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(payload_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + variant.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

} // namespace

FrameToFrameCoherenceExperimentOutput
run_frame_to_frame_coherence_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                        const FrameToFrameCoherenceExperimentConfig& config) {
    FrameToFrameCoherenceExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "frame-to-frame coherence experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "37_frame_to_frame_coherence.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for frame-to-frame coherence experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_elements =
        determine_logical_count(config.max_buffer_bytes, properties.limits.maxComputeWorkGroupCount[0]);
    if (max_elements == 0U) {
        std::cerr << "Scratch buffer too small for frame-to-frame coherence experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_elements
                  << ", source_span_bytes=" << compute_span_bytes(max_elements)
                  << ", output_span_bytes=" << compute_span_bytes(max_elements)
                  << ", frames_per_iteration=" << kFramesPerIteration
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_elements, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    auto* source_values = static_cast<uint32_t*>(buffers.source_mapped_ptr);
    if (source_values == nullptr || buffers.output_mapped_ptr == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for buffers.\n";
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }
    fill_source_values(source_values, max_elements);

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_elements);
    for (const auto& variant : kVariantDescriptors) {
        for (uint32_t element_count : problem_sizes) {
            if (!run_case(context, runner, pipeline, buffers, variant, element_count, output,
                          config.verbose_progress)) {
                output.all_points_correct = false;
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
