#include "experiments/persistent_threads_work_queues_experiment.hpp"

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

constexpr const char* kExperimentId = "40_persistent_threads_work_queues";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kDispatchCount = 1U;
constexpr uint32_t kTargetTaskCount = 67108864U;
constexpr uint32_t kMinimumTaskCount = kWorkgroupSize;
constexpr uint32_t kPersistentTargetWorkers = 32768U;
constexpr uint32_t kOutputSentinel = 0xFFFFFFFFU;

enum class SchedulingMode : uint32_t {
    StaticPartitioned = 0U,
    PersistentQueue = 1U,
};

enum class DistributionKind : std::uint8_t {
    UniformCost,
    SkewedTail,
};

struct SchedulingDescriptor {
    SchedulingMode mode;
    const char* name;
};

struct DistributionDescriptor {
    DistributionKind kind;
    const char* name;
};

constexpr std::array<SchedulingDescriptor, 2> kSchedulingDescriptors = {{
    {SchedulingMode::StaticPartitioned, "static_partitioned"},
    {SchedulingMode::PersistentQueue, "persistent_queue"},
}};

constexpr std::array<DistributionDescriptor, 2> kDistributionDescriptors = {{
    {DistributionKind::UniformCost, "uniform_cost"},
    {DistributionKind::SkewedTail, "skewed_tail"},
}};

struct BufferResources {
    BufferResource task_costs_buffer{};
    BufferResource output_buffer{};
    BufferResource queue_state_buffer{};
    void* task_costs_mapped_ptr = nullptr;
    void* output_mapped_ptr = nullptr;
    void* queue_state_mapped_ptr = nullptr;
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
    uint32_t task_count = 0U;
    uint32_t scheduling_mode = 0U;
    uint32_t reserved0 = 0U;
    uint32_t reserved1 = 0U;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

struct PreparedCaseData {
    std::vector<uint32_t> task_costs;
    std::vector<uint32_t> expected_outputs;
    double average_iterations = 0.0;
};

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

const char* scheduling_name(SchedulingMode mode) {
    for (const auto& descriptor : kSchedulingDescriptors) {
        if (descriptor.mode == mode) {
            return descriptor.name;
        }
    }
    return "unknown_schedule";
}

const char* distribution_name(DistributionKind kind) {
    for (const auto& descriptor : kDistributionDescriptors) {
        if (descriptor.kind == kind) {
            return descriptor.name;
        }
    }
    return "unknown_distribution";
}

std::string make_variant_name(SchedulingMode mode, DistributionKind kind) {
    return std::string(scheduling_name(mode)) + "_" + distribution_name(kind);
}

std::string make_case_name(SchedulingMode mode, DistributionKind kind, uint32_t task_count) {
    return std::string(kExperimentId) + "_" + make_variant_name(mode, kind) + "_tasks_" + std::to_string(task_count);
}

uint32_t execute_task(uint32_t task_index, uint32_t iteration_count) {
    uint32_t state = task_index * 1664525U + 1013904223U;
    for (uint32_t iter = 0U; iter < iteration_count; ++iter) {
        state ^= (state << 13U);
        state ^= (state >> 17U);
        state ^= (state << 5U);
        state = state * 1664525U + 1013904223U;
    }
    return state;
}

VkDeviceSize compute_task_costs_span_bytes(uint32_t task_count) {
    return static_cast<VkDeviceSize>(task_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_output_span_bytes(uint32_t task_count) {
    return static_cast<VkDeviceSize>(task_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

VkDeviceSize compute_queue_state_span_bytes() {
    return static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint32_t determine_task_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x) {
    const uint64_t task_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    const uint64_t capped = std::min({task_capacity, dispatch_capacity, static_cast<uint64_t>(kTargetTaskCount)});
    const uint64_t rounded = capped - (capped % kWorkgroupSize);
    if (rounded < kMinimumTaskCount) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_task_count) {
    const std::array<uint32_t, 6> base_sizes = {262144U, 1048576U, 4194304U, 16777216U, 33554432U, 67108864U};
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_task_count) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_task_count) {
        sizes.push_back(max_task_count);
    }
    return sizes;
}

float sample_unit(std::mt19937& generator) {
    return static_cast<float>(generator() & 0xFFFFU) / 65535.0F;
}

PreparedCaseData prepare_case_data(DistributionKind distribution, uint32_t task_count, uint32_t seed) {
    std::mt19937 generator(seed);
    PreparedCaseData prepared{};
    prepared.task_costs.resize(task_count, 0U);
    prepared.expected_outputs.resize(task_count, 0U);

    uint64_t iteration_total = 0U;
    for (uint32_t index = 0U; index < task_count; ++index) {
        uint32_t iterations = 0U;
        if (distribution == DistributionKind::UniformCost) {
            iterations = 96U + (generator() % 64U);
        } else {
            const float probability = sample_unit(generator);
            if (probability < 0.85F) {
                iterations = 8U + (generator() % 24U);
            } else {
                iterations = 96U + (generator() % 288U);
            }
        }

        prepared.task_costs[index] = iterations;
        prepared.expected_outputs[index] = execute_task(index, iterations);
        iteration_total += iterations;
    }

    prepared.average_iterations = task_count == 0U ? 0.0 : static_cast<double>(iteration_total) / task_count;
    return prepared;
}

uint64_t compute_logical_payload_bytes(uint32_t task_count) {
    return static_cast<uint64_t>(task_count) * sizeof(uint32_t) * 2ULL;
}

uint32_t determine_worker_threads(uint32_t task_count, uint32_t max_dispatch_groups_x) {
    const uint64_t dispatch_thread_cap = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    if (dispatch_thread_cap < kWorkgroupSize) {
        return 0U;
    }

    uint32_t workers = std::min(task_count, kPersistentTargetWorkers);
    workers = std::max(workers, kWorkgroupSize);
    const uint64_t worker_cap = dispatch_thread_cap - (dispatch_thread_cap % kWorkgroupSize);
    if (worker_cap == 0U) {
        return 0U;
    }
    workers = static_cast<uint32_t>(std::min<uint64_t>(workers, worker_cap));
    workers -= (workers % kWorkgroupSize);
    if (workers < kWorkgroupSize) {
        return 0U;
    }
    return workers;
}

uint64_t compute_estimated_global_traffic_bytes(uint32_t task_count, uint32_t worker_threads, SchedulingMode mode) {
    const uint64_t payload = compute_logical_payload_bytes(task_count);
    if (mode == SchedulingMode::StaticPartitioned) {
        return payload;
    }

    const uint64_t queue_atomic_ops = static_cast<uint64_t>(task_count) + worker_threads;
    return payload + (queue_atomic_ops * sizeof(uint32_t));
}

double compute_effective_gbps(uint32_t task_count, uint32_t worker_threads, SchedulingMode mode, double dispatch_ms) {
    return compute_effective_gbps_from_bytes(compute_estimated_global_traffic_bytes(task_count, worker_threads, mode),
                                             dispatch_ms);
}

bool create_buffer_resources(VulkanContext& context, uint32_t task_count, BufferResources& out_resources) {
    if (!create_buffer_resource(context.physical_device(), context.device(), compute_task_costs_span_bytes(task_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.task_costs_buffer)) {
        std::cerr << "Failed to create persistent-threads task-cost buffer.\n";
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_output_span_bytes(task_count),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.output_buffer)) {
        std::cerr << "Failed to create persistent-threads output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.task_costs_buffer);
        return false;
    }

    if (!create_buffer_resource(context.physical_device(), context.device(), compute_queue_state_span_bytes(),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                out_resources.queue_state_buffer)) {
        std::cerr << "Failed to create persistent-threads queue-state buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.task_costs_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.task_costs_buffer, "persistent-threads task-cost buffer",
                           out_resources.task_costs_mapped_ptr)) {
        destroy_buffer_resource(context.device(), out_resources.queue_state_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.task_costs_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.output_buffer, "persistent-threads output buffer",
                           out_resources.output_mapped_ptr)) {
        if (out_resources.task_costs_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.task_costs_buffer.memory);
            out_resources.task_costs_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.queue_state_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.task_costs_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.queue_state_buffer, "persistent-threads queue-state buffer",
                           out_resources.queue_state_mapped_ptr)) {
        if (out_resources.output_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.output_buffer.memory);
            out_resources.output_mapped_ptr = nullptr;
        }
        if (out_resources.task_costs_buffer.memory != VK_NULL_HANDLE) {
            vkUnmapMemory(context.device(), out_resources.task_costs_buffer.memory);
            out_resources.task_costs_mapped_ptr = nullptr;
        }
        destroy_buffer_resource(context.device(), out_resources.queue_state_buffer);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.task_costs_buffer);
        return false;
    }

    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.queue_state_mapped_ptr != nullptr && resources.queue_state_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.queue_state_buffer.memory);
        resources.queue_state_mapped_ptr = nullptr;
    }
    if (resources.output_mapped_ptr != nullptr && resources.output_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.output_buffer.memory);
        resources.output_mapped_ptr = nullptr;
    }
    if (resources.task_costs_mapped_ptr != nullptr && resources.task_costs_buffer.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.task_costs_buffer.memory);
        resources.task_costs_mapped_ptr = nullptr;
    }

    destroy_buffer_resource(context.device(), resources.queue_state_buffer);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.task_costs_buffer);
}

void update_descriptor_set(VulkanContext& context, const BufferResources& buffers, VkDescriptorSet descriptor_set) {
    const VkDescriptorBufferInfo task_costs_info{
        buffers.task_costs_buffer.buffer,
        0U,
        buffers.task_costs_buffer.size,
    };
    const VkDescriptorBufferInfo output_info{
        buffers.output_buffer.buffer,
        0U,
        buffers.output_buffer.size,
    };
    const VkDescriptorBufferInfo queue_state_info{
        buffers.queue_state_buffer.buffer,
        0U,
        buffers.queue_state_buffer.size,
    };

    // Shader binding contract:
    //   0 -> task-cost stream
    //   1 -> per-task output checksum
    //   2 -> single atomic queue cursor used by persistent mode
    VulkanComputeUtils::update_descriptor_set_buffers(context.device(), descriptor_set,
                                                      {
                                                          {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, task_costs_info},
                                                          {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, output_info},
                                                          {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, queue_state_info},
                                                      });
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, const BufferResources& buffers,
                               PipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load persistent-threads shader module: " << shader_path << "\n";
        return false;
    }

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create persistent-threads descriptor set layout.\n";
        return false;
    }

    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3U},
    };
    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create persistent-threads descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate persistent-threads descriptor set.\n";
        return false;
    }
    update_descriptor_set(context, buffers, out_resources.descriptor_set);

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {VK_SHADER_STAGE_COMPUTE_BIT, 0U, static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create persistent-threads pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create persistent-threads compute pipeline.\n";
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

bool validate_input_values(const uint32_t* task_costs, const std::vector<uint32_t>& expected, uint32_t task_count) {
    if (task_costs == nullptr || expected.size() < task_count) {
        return false;
    }
    return std::equal(expected.begin(), expected.begin() + task_count, task_costs);
}

bool validate_output_values(const uint32_t* output_values, const std::vector<uint32_t>& expected, uint32_t task_count) {
    if (output_values == nullptr || expected.size() < task_count) {
        return false;
    }
    return std::equal(expected.begin(), expected.begin() + task_count, output_values);
}

void reset_case_buffers(uint32_t* output_values, uint32_t* queue_state, uint32_t task_count) {
    std::fill_n(output_values, task_count, kOutputSentinel);
    queue_state[0] = 0U;
}

double run_dispatch(VulkanContext& context, const PipelineResources& pipeline_resources, const PushConstants& constants,
                    uint32_t worker_threads, uint32_t max_dispatch_groups_x) {
    uint32_t launched_threads = constants.task_count;
    // scheduling_mode changes the launch model: static mode launches one thread
    // per task, while persistent mode launches a fixed worker pool that drains
    // binding 2 with atomicAdd.
    if (static_cast<SchedulingMode>(constants.scheduling_mode) == SchedulingMode::PersistentQueue) {
        launched_threads = worker_threads;
    }

    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(launched_threads, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

void record_case_notes(std::string& notes, SchedulingMode mode, DistributionKind distribution, uint32_t task_count,
                       uint32_t worker_threads, const PreparedCaseData& prepared, uint64_t logical_payload_bytes,
                       uint64_t estimated_global_total_bytes, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "scheduling_mode=" + std::string(scheduling_name(mode)));
    append_note(notes, "distribution=" + std::string(distribution_name(distribution)));
    append_note(notes, "task_count=" + std::to_string(task_count));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "persistent_worker_threads=" + std::to_string(worker_threads));
    append_note(notes, "average_iterations_per_task=" + std::to_string(prepared.average_iterations));
    append_note(notes, "logical_payload_bytes=" + std::to_string(logical_payload_bytes));
    append_note(notes, "estimated_global_total_bytes=" + std::to_string(estimated_global_total_bytes));
    append_note(notes, "gbps_mode=estimated_global_bytes");
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner, const PipelineResources& pipeline_resources,
              const BufferResources& buffers, SchedulingMode mode, DistributionKind distribution, uint32_t task_count,
              uint32_t pattern_seed, uint32_t max_dispatch_groups_x,
              PersistentThreadsWorkQueuesExperimentOutput& output, bool verbose_progress) {
    auto* task_costs = static_cast<uint32_t*>(buffers.task_costs_mapped_ptr);
    auto* output_values = static_cast<uint32_t*>(buffers.output_mapped_ptr);
    auto* queue_state = static_cast<uint32_t*>(buffers.queue_state_mapped_ptr);
    if (task_costs == nullptr || output_values == nullptr || queue_state == nullptr) {
        std::cerr << "[" << kExperimentId
                  << "] Missing mapped buffers for variant=" << make_variant_name(mode, distribution) << ".\n";
        return false;
    }

    const PreparedCaseData prepared = prepare_case_data(distribution, task_count, pattern_seed);
    std::copy(prepared.task_costs.begin(), prepared.task_costs.begin() + task_count, task_costs);
    const bool input_valid = validate_input_values(task_costs, prepared.task_costs, task_count);

    const uint32_t worker_threads = determine_worker_threads(task_count, max_dispatch_groups_x);
    if (worker_threads == 0U) {
        std::cerr << "[" << kExperimentId << "] Failed to determine valid worker thread count.\n";
        return false;
    }

    // scheduling_mode values match the shader-side branch selection:
    //   0 = static_partitioned
    //   1 = persistent_queue
    const PushConstants constants{
        .task_count = task_count,
        .scheduling_mode = static_cast<uint32_t>(mode),
        .reserved0 = 0U,
        .reserved1 = 0U,
    };

    const uint64_t logical_payload_bytes = compute_logical_payload_bytes(task_count);
    const uint64_t estimated_global_total_bytes =
        compute_estimated_global_traffic_bytes(task_count, worker_threads, mode);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        reset_case_buffers(output_values, queue_state, task_count);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, constants, worker_threads, max_dispatch_groups_x);
        if (verbose_progress) {
            const bool dispatch_ok = std::isfinite(dispatch_ms);
            const bool correctness_pass = dispatch_ok && input_valid &&
                                          validate_output_values(output_values, prepared.expected_outputs, task_count);
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1U) << "/" << runner.warmup_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", task_count=" << task_count
                      << ", dispatch_ms=" << dispatch_ms << ", correctness=" << (correctness_pass ? "pass" : "fail")
                      << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        reset_case_buffers(output_values, queue_state, task_count);
        const double dispatch_ms =
            run_dispatch(context, pipeline_resources, constants, worker_threads, max_dispatch_groups_x);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool correctness_pass =
            dispatch_ok && input_valid && validate_output_values(output_values, prepared.expected_outputs, task_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, mode, distribution, task_count, worker_threads, prepared, logical_payload_bytes,
                          estimated_global_total_bytes, correctness_pass, dispatch_ok);

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1U) << "/" << runner.timed_iterations()
                      << " variant=" << make_variant_name(mode, distribution) << ", task_count=" << task_count
                      << ", dispatch_ms=" << dispatch_ms << ", end_to_end_ms=" << end_to_end_ms.count()
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = make_variant_name(mode, distribution),
            .problem_size = task_count,
            .dispatch_count = kDispatchCount,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(task_count, kDispatchCount, dispatch_ms),
            .gbps = compute_effective_gbps(task_count, worker_threads, mode, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });
    }

    output.summary_results.push_back(
        BenchmarkRunner::summarize_samples(make_case_name(mode, distribution, task_count), dispatch_samples));
    return true;
}

} // namespace

PersistentThreadsWorkQueuesExperimentOutput
run_persistent_threads_work_queues_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                              const PersistentThreadsWorkQueuesExperimentConfig& config) {
    PersistentThreadsWorkQueuesExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "persistent-threads work queues experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string shader_path =
        VulkanComputeUtils::resolve_shader_path(config.shader_path, "40_persistent_threads_work_queues.comp.spv");
    if (shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shader for persistent-threads work queues experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    const uint32_t max_dispatch_groups_x = properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_task_count = determine_task_count(config.max_buffer_bytes, max_dispatch_groups_x);
    if (max_task_count == 0U) {
        std::cerr << "Scratch buffer too small for persistent-threads work queues experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shader: " << shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_tasks=" << max_task_count
                  << ", task_costs_span_bytes=" << compute_task_costs_span_bytes(max_task_count)
                  << ", output_span_bytes=" << compute_output_span_bytes(max_task_count)
                  << ", queue_state_span_bytes=" << compute_queue_state_span_bytes()
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, max_task_count, buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    PipelineResources pipeline{};
    if (!create_pipeline_resources(context, shader_path, buffers, pipeline)) {
        destroy_pipeline_resources(context, pipeline);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_task_count);
    for (const auto& scheduling : kSchedulingDescriptors) {
        for (const auto& distribution : kDistributionDescriptors) {
            for (uint32_t task_count : problem_sizes) {
                const uint32_t seed = config.pattern_seed ^ (task_count * 0x9E3779B9U) ^
                                      (static_cast<uint32_t>(scheduling.mode) * 0x7F4A7C15U) ^
                                      (static_cast<uint32_t>(distribution.kind) * 0x85EBCA6BU);
                if (!run_case(context, runner, pipeline, buffers, scheduling.mode, distribution.kind, task_count, seed,
                              max_dispatch_groups_x, output, config.verbose_progress)) {
                    output.all_points_correct = false;
                }
            }
        }
    }

    destroy_pipeline_resources(context, pipeline);
    destroy_buffer_resources(context, buffers);
    return output;
}
