#include "experiments/histogram_atomic_contention_experiment.hpp"

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
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "23_histogram_atomic_contention";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kBinCount = 256U;
constexpr uint32_t kHotBin = 0U;
constexpr uint32_t kHotProbabilityMilli = 900U;
constexpr uint32_t kMixedHotsetCount = 8U;
constexpr uint32_t kMixedHotsetProbabilityMilli = 750U;

enum class ImplementationKind : std::uint8_t {
    GlobalAtomics,
    PrivatizedShared,
};

enum class DistributionKind : std::uint8_t {
    Uniform,
    HotBin90,
    MixedHotset75,
};

struct ImplementationDescriptor {
    ImplementationKind kind;
    const char* implementation_name;
    const char* shader_filename;
};

struct DistributionDescriptor {
    DistributionKind kind;
    const char* distribution_name;
};

constexpr std::array<ImplementationDescriptor, 2> kImplementationDescriptors = {{
    {ImplementationKind::GlobalAtomics, "global_atomics", "23_histogram_global_atomics.comp.spv"},
    {ImplementationKind::PrivatizedShared, "privatized_shared", "23_histogram_privatized.comp.spv"},
}};

constexpr std::array<DistributionDescriptor, 3> kDistributionDescriptors = {{
    {DistributionKind::Uniform, "uniform"},
    {DistributionKind::HotBin90, "hot_bin_90"},
    {DistributionKind::MixedHotset75, "mixed_hotset_75"},
}};

struct BufferResources {
    BufferResource input_buffer{};
    BufferResource histogram_buffer{};
    void* input_mapped_ptr = nullptr;
    void* histogram_mapped_ptr = nullptr;
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

const char* implementation_name(ImplementationKind implementation_kind) {
    for (const auto& descriptor : kImplementationDescriptors) {
        if (descriptor.kind == implementation_kind) {
            return descriptor.implementation_name;
        }
    }
    return "unknown";
}

const char* distribution_name(DistributionKind distribution_kind) {
    for (const auto& descriptor : kDistributionDescriptors) {
        if (descriptor.kind == distribution_kind) {
            return descriptor.distribution_name;
        }
    }
    return "unknown";
}

std::string make_variant_name(ImplementationKind implementation_kind, DistributionKind distribution_kind) {
    return std::string(implementation_name(implementation_kind)) + "_" + distribution_name(distribution_kind);
}

std::string make_case_name(ImplementationKind implementation_kind, DistributionKind distribution_kind,
                           uint32_t sample_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(implementation_kind, distribution_kind) + "_samples_" +
           std::to_string(sample_count);
}

uint32_t determine_sample_count(std::size_t total_budget_bytes, uint32_t max_dispatch_groups_x) {
    const VkDeviceSize histogram_bytes =
        static_cast<VkDeviceSize>(kBinCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
    if (total_budget_bytes <= histogram_bytes) {
        return 0U;
    }

    const uint64_t input_budget_bytes = static_cast<uint64_t>(total_budget_bytes - histogram_bytes);
    const uint64_t buffer_limited_count = input_budget_bytes / sizeof(uint32_t);
    const uint64_t dispatch_limited_count = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t effective_count = std::min(
        {buffer_limited_count, dispatch_limited_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});

    return static_cast<uint32_t>(effective_count);
}

VkDeviceSize compute_input_span_bytes(uint32_t sample_count) {
    return static_cast<VkDeviceSize>(sample_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_histogram_span_bytes() {
    return static_cast<VkDeviceSize>(kBinCount) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t hot_probability_milli(DistributionKind distribution_kind) {
    switch (distribution_kind) {
    case DistributionKind::Uniform:
        return 0U;
    case DistributionKind::HotBin90:
        return kHotProbabilityMilli;
    case DistributionKind::MixedHotset75:
        return kMixedHotsetProbabilityMilli;
    }

    return 0U;
}

uint32_t hotset_count(DistributionKind distribution_kind) {
    switch (distribution_kind) {
    case DistributionKind::Uniform:
        return 0U;
    case DistributionKind::HotBin90:
        return 1U;
    case DistributionKind::MixedHotset75:
        return kMixedHotsetCount;
    }

    return 0U;
}

uint64_t compute_estimated_global_histogram_bytes(ImplementationKind implementation_kind,
                                                  const std::vector<uint32_t>& input_values) {
    if (implementation_kind == ImplementationKind::GlobalAtomics) {
        return static_cast<uint64_t>(input_values.size()) * sizeof(uint32_t);
    }

    uint64_t nonzero_bin_flushes = 0U;
    for (std::size_t group_base = 0; group_base < input_values.size(); group_base += kWorkgroupSize) {
        std::array<bool, kBinCount> touched_bins{};
        const std::size_t group_end =
            std::min(group_base + static_cast<std::size_t>(kWorkgroupSize), input_values.size());
        for (std::size_t index = group_base; index < group_end; ++index) {
            touched_bins[input_values[index]] = true;
        }
        nonzero_bin_flushes += static_cast<uint64_t>(std::count(touched_bins.begin(), touched_bins.end(), true));
    }

    return nonzero_bin_flushes * sizeof(uint32_t);
}

double compute_effective_gbps(uint32_t sample_count, uint64_t estimated_global_histogram_bytes, double dispatch_ms) {
    if (!std::isfinite(dispatch_ms) || dispatch_ms <= 0.0) {
        return 0.0;
    }

    const uint64_t input_bytes = static_cast<uint64_t>(sample_count) * sizeof(uint32_t);
    return compute_effective_gbps_from_bytes(input_bytes + estimated_global_histogram_bytes, dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize input_bytes, BufferResources& out_resources) {
    if (!create_buffer_resource(
            context.physical_device(), context.device(), input_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.input_buffer)) {
        std::cerr << "Failed to create histogram input buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_histogram_span_bytes(),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.histogram_buffer)) {
        std::cerr << "Failed to create histogram output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.input_buffer, "histogram input buffer",
                           out_resources.input_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.histogram_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.histogram_buffer, "histogram output buffer",
                           out_resources.histogram_mapped_ptr)) {
        if (out_resources.input_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.input_buffer.memory);
            out_resources.input_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.histogram_buffer);
        destroy_buffer_resource(context.device(), out_resources.input_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.histogram_mapped_ptr != nullptr && resources.histogram_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.histogram_buffer.memory);
        resources.histogram_mapped_ptr = nullptr;
    }
    if (resources.input_mapped_ptr != nullptr && resources.input_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.input_buffer.memory);
        resources.input_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.histogram_buffer);
    destroy_buffer_resource(context.device(), resources.input_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo input_info{buffers.input_buffer.buffer, 0U, buffers.input_buffer.size};
    const VkDescriptorBufferInfo histogram_info{buffers.histogram_buffer.buffer, 0U, buffers.histogram_buffer.size};

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
                                                              .buffer_info = histogram_info,
                                                          },
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load histogram atomic contention shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        VkDescriptorSetLayoutBinding{0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create histogram atomic contention descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create histogram atomic contention descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate histogram atomic contention descriptor set.\n";
        return false;
    }

    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(uint32_t))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create histogram atomic contention pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create histogram atomic contention compute pipeline.\n";
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

double run_dispatch(VulkanContext& context, const PipelineResources& resources, uint32_t sample_count) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(sample_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const uint32_t push_count = sample_count;
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipeline_layout, 0U, 1U,
                                &resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(push_count)), &push_count);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void fill_histogram_zeroes(uint32_t* histogram_values) {
    std::fill_n(histogram_values, kBinCount, 0U);
}

std::vector<uint32_t> build_input_values(DistributionKind distribution_kind, uint32_t sample_count, uint32_t seed) {
    std::vector<uint32_t> values(sample_count, 0U);
    std::mt19937 engine(seed);

    for (uint32_t index = 0U; index < sample_count; ++index) {
        const uint32_t draw = engine();
        switch (distribution_kind) {
        case DistributionKind::Uniform:
            values[index] = draw % kBinCount;
            break;
        case DistributionKind::HotBin90:
            values[index] = (draw % 1000U) < kHotProbabilityMilli ? kHotBin : ((draw >> 10U) % kBinCount);
            break;
        case DistributionKind::MixedHotset75:
            values[index] = (draw % 1000U) < kMixedHotsetProbabilityMilli ? ((draw >> 8U) % kMixedHotsetCount)
                                                                          : ((draw >> 16U) % kBinCount);
            break;
        }
    }

    return values;
}

std::array<uint32_t, kBinCount> build_reference_histogram(const std::vector<uint32_t>& input_values) {
    std::array<uint32_t, kBinCount> histogram{};
    histogram.fill(0U);
    for (const uint32_t value : input_values) {
        ++histogram[value];
    }
    return histogram;
}

bool validate_input_values(const uint32_t* input_values, const std::vector<uint32_t>& reference_values) {
    for (std::size_t index = 0; index < reference_values.size(); ++index) {
        if (input_values[index] != reference_values[index]) {
            return false;
        }
    }
    return true;
}

bool validate_histogram_values(const uint32_t* histogram_values,
                               const std::array<uint32_t, kBinCount>& reference_histogram) {
    for (uint32_t bin_index = 0U; bin_index < kBinCount; ++bin_index) {
        if (histogram_values[bin_index] != reference_histogram[bin_index]) {
            return false;
        }
    }
    return true;
}

uint32_t distribution_seed(const HistogramAtomicContentionExperimentConfig& config,
                           DistributionKind distribution_kind) {
    switch (distribution_kind) {
    case DistributionKind::Uniform:
        return config.pattern_seed ^ 0x13579BDFU;
    case DistributionKind::HotBin90:
        return config.pattern_seed ^ 0x2468ACE0U;
    case DistributionKind::MixedHotset75:
        return config.pattern_seed ^ 0x55AA33CCU;
    }
    return config.pattern_seed;
}

void record_case_notes(std::string& notes, ImplementationKind implementation_kind, DistributionKind distribution_kind,
                       const HistogramAtomicContentionExperimentConfig& config, uint32_t sample_count,
                       uint32_t group_count_x, uint64_t estimated_global_histogram_bytes, bool correctness_pass,
                       bool dispatch_ok) {
    const uint64_t input_span_bytes = static_cast<uint64_t>(compute_input_span_bytes(sample_count));
    const uint64_t histogram_span_bytes = static_cast<uint64_t>(compute_histogram_span_bytes());
    const uint64_t estimated_global_input_bytes = static_cast<uint64_t>(sample_count) * sizeof(uint32_t);

    append_note(notes, std::string("implementation=") + implementation_name(implementation_kind));
    append_note(notes, std::string("distribution=") + distribution_name(distribution_kind));
    append_note(notes, "seed=" + std::to_string(distribution_seed(config, distribution_kind)));
    append_note(notes, "bin_count=" + std::to_string(kBinCount));
    append_note(notes, "hot_probability_milli=" + std::to_string(hot_probability_milli(distribution_kind)));
    append_note(notes, "hotset_count=" + std::to_string(hotset_count(distribution_kind)));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "group_count_x=" + std::to_string(group_count_x));
    append_note(notes, "sample_count=" + std::to_string(sample_count));
    append_note(notes, "input_span_bytes=" + std::to_string(input_span_bytes));
    append_note(notes, "histogram_span_bytes=" + std::to_string(histogram_span_bytes));
    append_note(notes,
                "scratch_size_bytes=" + std::to_string(static_cast<unsigned long long>(config.max_buffer_bytes)));
    append_note(notes, "estimated_global_input_bytes=" + std::to_string(estimated_global_input_bytes));
    append_note(notes, "estimated_global_histogram_bytes=" + std::to_string(estimated_global_histogram_bytes));
    append_note(notes, "estimated_global_total_bytes=" +
                           std::to_string(estimated_global_input_bytes + estimated_global_histogram_bytes));
    append_note(notes, "barriers_per_workgroup=" +
                           std::to_string(implementation_kind == ImplementationKind::PrivatizedShared ? 2U : 0U));
    append_note(notes, "validation_mode=exact_histogram");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const BufferResources& buffers,
              const PipelineResources& pipeline_resources, ImplementationKind implementation_kind,
              DistributionKind distribution_kind, const HistogramAtomicContentionExperimentConfig& config,
              uint32_t sample_count, HistogramAtomicContentionExperimentOutput& output) {
    const std::string variant_name = make_variant_name(implementation_kind, distribution_kind);
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(sample_count, kWorkgroupSize);
    if (group_count_x == 0U) {
        std::cerr << "[" << kExperimentId << "] Unable to compute a legal dispatch size for variant " << variant_name
                  << ".\n";
        return false;
    }

    const std::vector<uint32_t> input_reference =
        build_input_values(distribution_kind, sample_count, distribution_seed(config, distribution_kind));
    const std::array<uint32_t, kBinCount> histogram_reference = build_reference_histogram(input_reference);
    const uint64_t estimated_global_histogram_bytes =
        compute_estimated_global_histogram_bytes(implementation_kind, input_reference);

    auto* input_values = static_cast<uint32_t*>(buffers.input_mapped_ptr);
    auto* histogram_values = static_cast<uint32_t*>(buffers.histogram_mapped_ptr);
    if (input_values == nullptr || histogram_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped pointers for variant " << variant_name << ".\n";
        return false;
    }

    std::copy(input_reference.begin(), input_reference.end(), input_values);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case start: variant=" << variant_name
                  << ", sample_count=" << sample_count << ", group_count_x=" << group_count_x
                  << ", input_span_bytes=" << compute_input_span_bytes(sample_count)
                  << ", histogram_span_bytes=" << compute_histogram_span_bytes()
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        fill_histogram_zeroes(histogram_values);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, sample_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, input_reference) &&
                             validate_histogram_values(histogram_values, histogram_reference);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", correctness=" << ((dispatch_ok && data_ok) ? "pass" : "fail") << "\n";
        }

        if (!dispatch_ok || !data_ok) {
            std::cerr << "[" << kExperimentId << "] Warmup issue for variant=" << variant_name
                      << ", sample_count=" << sample_count << ", dispatch_ok=" << (dispatch_ok ? "true" : "false")
                      << ", data_ok=" << (data_ok ? "true" : "false") << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();

        fill_histogram_zeroes(histogram_values);
        const double dispatch_ms = run_dispatch(context, pipeline_resources, sample_count);
        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool data_ok = dispatch_ok && validate_input_values(input_values, input_reference) &&
                             validate_histogram_values(histogram_values, histogram_reference);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool correctness_pass = dispatch_ok && data_ok;
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, implementation_kind, distribution_kind, config, sample_count, group_count_x,
                          estimated_global_histogram_bytes, correctness_pass, dispatch_ok);

        if (config.verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << variant_name << ", dispatch_ms=" << dispatch_ms
                      << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = variant_name,
            .problem_size = sample_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(sample_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(sample_count, estimated_global_histogram_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    const BenchmarkResult summary = BenchmarkRunner::summarize_samples(
        make_case_name(implementation_kind, distribution_kind, sample_count), dispatch_samples);
    output.summary_results.push_back(summary);

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Case complete: variant=" << variant_name
                  << ", sample_count=" << sample_count << ", samples=" << summary.sample_count
                  << ", median_gpu_ms=" << summary.median_ms << "\n";
    }

    return true;
}

} // namespace

HistogramAtomicContentionExperimentOutput
run_histogram_atomic_contention_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                           const HistogramAtomicContentionExperimentConfig& config) {
    HistogramAtomicContentionExperimentOutput output{};

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "histogram atomic contention experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    std::array<std::string, kImplementationDescriptors.size()> shader_paths;
    for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
        const auto& descriptor = kImplementationDescriptors[index];
        const std::string user_path = descriptor.kind == ImplementationKind::GlobalAtomics
                                          ? config.global_shader_path
                                          : config.privatized_shader_path;
        shader_paths[index] = VulkanComputeUtils::resolve_shader_path(user_path, descriptor.shader_filename);
        if (shader_paths[index].empty()) {
            std::cerr << "Could not locate SPIR-V shader for histogram atomic contention variant "
                      << descriptor.implementation_name << ".\n";
            output.all_points_correct = false;
            return output;
        }
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);

    const uint32_t sample_count =
        determine_sample_count(config.max_buffer_bytes, device_properties.limits.maxComputeWorkGroupCount[0]);
    if (sample_count < kWorkgroupSize) {
        std::cerr << "Scratch buffer too small for histogram atomic contention experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] Global shader: " << shader_paths[0] << "\n";
        std::cout << "[" << kExperimentId << "] Privatized shader: " << shader_paths[1] << "\n";
        std::cout << "[" << kExperimentId << "] sample_count=" << sample_count
                  << ", input_span_bytes=" << compute_input_span_bytes(sample_count)
                  << ", histogram_span_bytes=" << compute_histogram_span_bytes()
                  << ", scratch_size_bytes=" << config.max_buffer_bytes
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_input_span_bytes(sample_count), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    std::array<PipelineResources, kImplementationDescriptors.size()> pipeline_resources{};
    for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
        if (!create_pipeline_resources(context, shader_paths[index], buffers, pipeline_resources[index])) {
            for (PipelineResources& resources : pipeline_resources) {
                destroy_pipeline_resources(context, resources);
            }
            destroy_buffer_resources(context, buffers);
            output.all_points_correct = false;
            return output;
        }
    }

    for (const auto& distribution_descriptor : kDistributionDescriptors) {
        for (std::size_t index = 0; index < kImplementationDescriptors.size(); ++index) {
            const auto& implementation_descriptor = kImplementationDescriptors[index];
            if (!run_case(context, runner, buffers, pipeline_resources[index], implementation_descriptor.kind,
                          distribution_descriptor.kind, config, sample_count, output)) {
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
