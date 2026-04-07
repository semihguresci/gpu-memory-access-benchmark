#include "experiments/gather_access_pattern_experiment.hpp"

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
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "12_gather_access_pattern";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kSourceBaseValue = 0x01020304U;
constexpr uint32_t kSentinelValue = 0xA5A5A5A5U;
constexpr VkDeviceSize kBytesPerLogicalElement = static_cast<VkDeviceSize>(sizeof(uint32_t) * 3U);

enum class GatherDistributionKind : std::uint8_t {
    kIdentity,
    kBlockCoherent,
    kClusteredRandom,
    kRandomPermutation,
};

struct GatherCaseDescriptor {
    GatherDistributionKind kind;
};

constexpr std::array<GatherCaseDescriptor, 4> kCaseDescriptors = {{
    {GatherDistributionKind::kIdentity},
    {GatherDistributionKind::kBlockCoherent},
    {GatherDistributionKind::kClusteredRandom},
    {GatherDistributionKind::kRandomPermutation},
}};

struct CaseBufferResources {
    BufferResource src_buffer{};
    BufferResource index_buffer{};
    BufferResource dst_buffer{};
    void* src_mapped_ptr = nullptr;
    void* index_mapped_ptr = nullptr;
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

std::string make_variant_name(GatherDistributionKind kind, const GatherAccessPatternExperimentConfig& config) {
    switch (kind) {
    case GatherDistributionKind::kIdentity:
        return "identity";
    case GatherDistributionKind::kBlockCoherent:
        return "block_coherent_" + std::to_string(config.block_size);
    case GatherDistributionKind::kClusteredRandom:
        return "clustered_random_" + std::to_string(config.cluster_size);
    case GatherDistributionKind::kRandomPermutation:
        return "random_permutation";
    }

    return "unknown";
}

std::string make_case_name(const std::string& variant_name, uint32_t logical_count) {
    return std::string(kExperimentId) + "_" + variant_name + "_elements_" + std::to_string(logical_count);
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

bool validate_case_values(const uint32_t* src_values, const uint32_t* index_values, const uint32_t* dst_values,
                          const std::vector<uint32_t>& index_reference, uint32_t element_count) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        const uint32_t source_index = index_reference[index];
        if (source_index >= element_count) {
            return false;
        }

        if (index_values[index] != source_index) {
            return false;
        }

        const uint32_t expected_src = source_pattern_value(source_index);
        if (src_values[source_index] != expected_src) {
            return false;
        }

        if (dst_values[index] != (expected_src + 1U)) {
            return false;
        }
    }

    return true;
}

uint32_t determine_logical_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t buffer_limited_count = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count_u64 = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    return static_cast<uint32_t>(effective_count_u64);
}

VkDeviceSize compute_physical_span_bytes(uint32_t logical_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_logical_bytes(uint32_t logical_count, uint32_t dispatch_count) {
    return static_cast<VkDeviceSize>(logical_count) * static_cast<VkDeviceSize>(dispatch_count) *
           kBytesPerLogicalElement;
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
        std::cerr << "Failed to create gather source buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.index_buffer)) {
        std::cerr << "Failed to create gather index buffer.\n";
        return false;
    }

    if (!create_buffer_resource(
            context.physical_device(), context.device(), buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.dst_buffer)) {
        std::cerr << "Failed to create gather destination buffer.\n";
        return false;
    }

    if (!map_buffer_memory(context, out_resources.src_buffer, "gather source buffer", out_resources.src_mapped_ptr)) {
        return false;
    }

    if (!map_buffer_memory(context, out_resources.index_buffer, "gather index buffer",
                           out_resources.index_mapped_ptr)) {
        return false;
    }

    if (!map_buffer_memory(context, out_resources.dst_buffer, "gather destination buffer",
                           out_resources.dst_mapped_ptr)) {
        return false;
    }

    return true;
}

void destroy_case_buffer_resources(VulkanContext& context, CaseBufferResources& resources) {
    if (resources.dst_mapped_ptr != nullptr && resources.dst_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.dst_buffer.memory);
        resources.dst_mapped_ptr = nullptr;
    }

    if (resources.index_mapped_ptr != nullptr && resources.index_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.index_buffer.memory);
        resources.index_mapped_ptr = nullptr;
    }

    if (resources.src_mapped_ptr != nullptr && resources.src_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.src_buffer.memory);
        resources.src_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.dst_buffer);
    destroy_buffer_resource(context.device(), resources.index_buffer);
    destroy_buffer_resource(context.device(), resources.src_buffer);
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load gather access pattern shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create gather access pattern descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create gather access pattern descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate gather access pattern descriptor set.\n";
        return false;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(uint32_t))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create gather access pattern pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create gather access pattern compute pipeline.\n";
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
    const VkDescriptorBufferInfo index_info{
        buffers.index_buffer.buffer,
        0U,
        buffers.index_buffer.size,
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
                                                              .buffer_info = index_info,
                                                          },
                                                          VulkanComputeUtils::DescriptorBufferBindingUpdate{
                                                              .binding = 2U,
                                                              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                              .buffer_info = dst_info,
                                                          },
                                                      });
}

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t logical_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const uint32_t push_count = logical_count;
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_count)), &push_count);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

std::vector<uint32_t> make_index_reference(GatherDistributionKind kind, uint32_t logical_count,
                                           const GatherAccessPatternExperimentConfig& config) {
    std::vector<uint32_t> indices(logical_count);
    std::iota(indices.begin(), indices.end(), 0U);

    std::mt19937 engine(config.pattern_seed);
    switch (kind) {
    case GatherDistributionKind::kIdentity:
        return indices;
    case GatherDistributionKind::kBlockCoherent: {
        std::vector<uint32_t> block_starts;
        const uint32_t block_size = std::max(1U, config.block_size);
        for (uint32_t block_start = 0U; block_start < logical_count; block_start += block_size) {
            block_starts.push_back(block_start);
        }

        std::shuffle(block_starts.begin(), block_starts.end(), engine);

        std::size_t output_index = 0U;
        for (const uint32_t block_start : block_starts) {
            const uint32_t block_end = std::min(block_start + block_size, logical_count);
            for (uint32_t index = block_start; index < block_end; ++index) {
                indices[output_index++] = index;
            }
        }
        return indices;
    }
    case GatherDistributionKind::kClusteredRandom: {
        const uint32_t cluster_size = std::max(1U, config.cluster_size);
        std::vector<uint32_t> cluster_starts;
        for (uint32_t cluster_start = 0U; cluster_start < logical_count; cluster_start += cluster_size) {
            cluster_starts.push_back(cluster_start);
        }

        std::shuffle(cluster_starts.begin(), cluster_starts.end(), engine);

        std::size_t output_index = 0U;
        for (const uint32_t cluster_start : cluster_starts) {
            const uint32_t cluster_end = std::min(cluster_start + cluster_size, logical_count);
            std::vector<uint32_t> cluster_indices;
            cluster_indices.reserve(cluster_end - cluster_start);
            for (uint32_t index = cluster_start; index < cluster_end; ++index) {
                cluster_indices.push_back(index);
            }

            std::shuffle(cluster_indices.begin(), cluster_indices.end(), engine);
            for (const uint32_t index : cluster_indices) {
                indices[output_index++] = index;
            }
        }
        return indices;
    }
    case GatherDistributionKind::kRandomPermutation:
        std::shuffle(indices.begin(), indices.end(), engine);
        return indices;
    }

    return indices;
}

void record_case_run(std::string& notes, const std::string& variant_name,
                     const GatherAccessPatternExperimentConfig& config, uint32_t logical_count,
                     VkDeviceSize physical_span_bytes, VkDeviceSize scratch_size_bytes, uint32_t group_count_x,
                     bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "distribution=" + variant_name);
    append_note(notes, "seed=" + std::to_string(config.pattern_seed));
    append_note(notes, "block_size=" + std::to_string(config.block_size));
    append_note(notes, "cluster_size=" + std::to_string(config.cluster_size));
    append_note(notes, "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(scratch_size_bytes)));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "dispatch_count=" + std::to_string(kDispatchCount));
    append_note(notes, "logical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_elements=" + std::to_string(logical_count));
    append_note(notes, "physical_span_bytes=" + std::to_string(static_cast<unsigned long long>(physical_span_bytes)));
    append_note(notes, "allocated_span_bytes=" + std::to_string(static_cast<unsigned long long>(physical_span_bytes)));
    append_note(notes, "bytes_per_logical_element=" +
                           std::to_string(static_cast<unsigned long long>(kBytesPerLogicalElement)));
    append_note(notes, "validation_mode=exact_uint32");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const GatherAccessPatternExperimentConfig& config, GatherDistributionKind kind, uint32_t logical_count,
              VkDeviceSize physical_span_bytes, GatherAccessPatternExperimentOutput& output) {
    const std::string variant_name = make_variant_name(kind, config);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(logical_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for variant " << variant_name
                  << ".\n";
        return false;
    }

    std::vector<uint32_t> index_reference = make_index_reference(kind, logical_count, config);
    if (index_reference.size() != logical_count) {
        std::cerr << "[" << kExperimentId << "] Failed to build index reference for variant " << variant_name << ".\n";
        return false;
    }

    CaseBufferResources buffers{};
    if (!create_case_buffer_resources(context, physical_span_bytes, buffers)) {
        destroy_case_buffer_resources(context, buffers);
        return false;
    }

    update_case_descriptor_set(context, pipeline_resources, buffers);

    auto* src_values = static_cast<uint32_t*>(buffers.src_mapped_ptr);
    auto* index_values = static_cast<uint32_t*>(buffers.index_mapped_ptr);
    auto* dst_values = static_cast<uint32_t*>(buffers.dst_mapped_ptr);
    if (src_values == nullptr || index_values == nullptr || dst_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for variant " << variant_name << ".\n";
        destroy_case_buffer_resources(context, buffers);
        return false;
    }

    fill_source_values(src_values, logical_count);
    std::ranges::copy(index_reference, index_values);

    const std::size_t timed_iterations = static_cast<std::size_t>(std::max(0, runner.timed_iterations()));
    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(timed_iterations);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", physical_span_bytes=" << physical_span_bytes
                  << ", group_count_x=" << group_count_x << ", seed=" << config.pattern_seed
                  << ", block_size=" << config.block_size << ", cluster_size=" << config.cluster_size
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_sentinel_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_case_values(src_values, index_values, dst_values, index_reference, logical_count);
        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }
        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", logical_elements=" << logical_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_sentinel_values(dst_values, logical_count);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, logical_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok =
            dispatch_ok && validate_case_values(src_values, index_values, dst_values, index_reference, logical_count);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_run(notes, variant_name, config, logical_count, physical_span_bytes,
                        static_cast<VkDeviceSize>(config.max_buffer_bytes), group_count_x, correctness_pass,
                        dispatch_ok);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

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
        BenchmarkRunner::summarize_samples(make_case_name(variant_name, logical_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", logical_elements=" << logical_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    destroy_case_buffer_resources(context, buffers);
    return true;
}

} // namespace

GatherAccessPatternExperimentOutput
run_gather_access_pattern_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                     const GatherAccessPatternExperimentConfig& config) {
    GatherAccessPatternExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "gather access pattern experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.block_size == 0U || config.cluster_size == 0U) {
        std::cerr << "gather access pattern experiment requires non-zero block and cluster sizes.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "12_gather_access_pattern.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for gather access pattern experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t logical_count =
        determine_logical_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (logical_count == 0U) {
        std::cerr << "Scratch buffer too small for gather access pattern experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        const VkDeviceSize physical_span_bytes = compute_physical_span_bytes(logical_count);
        std::cout << "[" << kExperimentId << "] Shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] logical_elements=" << logical_count
                  << ", physical_span_bytes_per_buffer=" << physical_span_bytes
                  << ", scratch_size_bytes=" << config.max_buffer_bytes << ", seed=" << config.pattern_seed
                  << ", block_size=" << config.block_size << ", cluster_size=" << config.cluster_size
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    PipelineResources pipeline_resources{};
    if (!create_pipeline_resources(context, shader_path, pipeline_resources)) {
        destroy_pipeline_resources(context, pipeline_resources);
        output.all_points_correct = false;
        return output;
    }

    const VkDeviceSize physical_span_bytes = compute_physical_span_bytes(logical_count);
    for (const GatherCaseDescriptor& case_descriptor : kCaseDescriptors) {
        if (!run_case(context, runner, pipeline_resources, config, case_descriptor.kind, logical_count,
                      physical_span_bytes, output)) {
            output.all_points_correct = false;
        }
    }

    destroy_pipeline_resources(context, pipeline_resources);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Finished run: summaries=" << output.summary_results.size()
                  << ", rows=" << output.rows.size()
                  << ", all_points_correct=" << (output.all_points_correct ? "true" : "false") << "\n";
    }

    return output;
}
