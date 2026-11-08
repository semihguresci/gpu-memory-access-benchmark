#include "experiments/buffer_vs_image_representation_experiment.hpp"

#include "utils/buffer_utils.hpp"
#include "utils/experiment_metrics.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using ExperimentMetrics::compute_effective_gbps_from_bytes;
using ExperimentMetrics::compute_throughput_elements_per_second;

constexpr const char* kExperimentId = "48_buffer_vs_image_representation";
constexpr uint32_t kWorkgroupSize = 256U;
constexpr uint32_t kFetchedTexelsPerInvocation = 4U;
// Large enough to define interesting 2D rectangles and small enough that every
// problem size is an exact multiple of it, so W * H always tiles exactly.
constexpr uint32_t kMaxGridWidth = 1024U;
constexpr uint32_t kAccessNeighborhoodClamped = 0U;
constexpr uint32_t kAccessRandomScatter = 1U;

uint32_t hash32(uint32_t value) {
    // Mirrors hash32() in every 48_*.comp shader.
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

uint32_t mix_state(uint32_t state, uint32_t value) {
    // Mirrors mix_state() in every 48_*.comp shader.
    return (state ^ value) * 2654435761U;
}

// Dataset content is a function of the logical 2D coordinate only, so all four
// representations hold identical values at identical coordinates.
uint32_t dataset_value(uint32_t x, uint32_t y) {
    return hash32((y * 747796405U) ^ (x * 2654435761U));
}

uint32_t hashed_fetch_x(uint32_t id, uint32_t k, uint32_t grid_width) {
    return hash32(((id * kFetchedTexelsPerInvocation) + k) * 2U) % grid_width;
}

uint32_t hashed_fetch_y(uint32_t id, uint32_t k, uint32_t grid_height) {
    return hash32((((id * kFetchedTexelsPerInvocation) + k) * 2U) + 1U) % grid_height;
}

void fetch_coordinates(uint32_t id, uint32_t grid_width, uint32_t grid_height, uint32_t access_mode, uint32_t k,
                       uint32_t& out_x, uint32_t& out_y) {
    const uint32_t x0 = id % grid_width;
    const uint32_t y0 = id / grid_width;
    if (access_mode == kAccessNeighborhoodClamped) {
        const uint32_t dx = (k == 1U || k == 3U) ? 1U : 0U;
        const uint32_t dy = (k >= 2U) ? 1U : 0U;
        out_x = std::min(x0 + dx, grid_width - 1U);
        out_y = std::min(y0 + dy, grid_height - 1U);
        return;
    }
    out_x = hashed_fetch_x(id, k, grid_width);
    out_y = hashed_fetch_y(id, k, grid_height);
}

enum class RepresentationKind : uint8_t { kRowMajorBuffer, kColumnMajorBuffer, kStorageImage, kSampledImage };

std::string representation_name(RepresentationKind kind) {
    switch (kind) {
    case RepresentationKind::kRowMajorBuffer:
        return "row_major_buffer";
    case RepresentationKind::kColumnMajorBuffer:
        return "column_major_buffer";
    case RepresentationKind::kStorageImage:
        return "storage_image";
    case RepresentationKind::kSampledImage:
        return "sampled_image";
    }
    return "unknown";
}

struct VariantDescriptor {
    RepresentationKind kind = RepresentationKind::kRowMajorBuffer;
    uint32_t access_mode = kAccessNeighborhoodClamped;
    std::string variant_name;
};

std::vector<VariantDescriptor> build_variant_descriptors() {
    static constexpr std::array<std::pair<uint32_t, const char*>, 2> kAccessModes = {
        {{kAccessNeighborhoodClamped, "neighborhood_clamped"}, {kAccessRandomScatter, "random_scatter"}},
    };
    static constexpr std::array<RepresentationKind, 4> kKinds = {
        RepresentationKind::kRowMajorBuffer,
        RepresentationKind::kColumnMajorBuffer,
        RepresentationKind::kStorageImage,
        RepresentationKind::kSampledImage,
    };

    std::vector<VariantDescriptor> variants;
    variants.reserve(kKinds.size() * kAccessModes.size());
    for (const RepresentationKind kind : kKinds) {
        for (const auto& [mode, token] : kAccessModes) {
            variants.push_back(VariantDescriptor{
                .kind = kind,
                .access_mode = mode,
                .variant_name = representation_name(kind) + "_" + token,
            });
        }
    }
    return variants;
}

struct BufferResources {
    BufferResource row_input_buffer{};
    BufferResource column_input_buffer{};
    BufferResource output_buffer{};
    BufferResource staging_in{};
    BufferResource staging_out{};
    void* staging_in_mapped_ptr = nullptr;
    void* staging_out_mapped_ptr = nullptr;
};

struct ImageResource {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceSize allocation_size = 0;
    uint32_t memory_type_index = 0;
    uint32_t heap_index = 0;
};

struct SamplerResource {
    VkSampler sampler = VK_NULL_HANDLE;
};

struct RepresentationPipelineResources {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
    uint32_t element_count = 0U;
    uint32_t grid_width = 0U;
    uint32_t grid_height = 0U;
    uint32_t access_mode = kAccessNeighborhoodClamped;
};

static_assert(sizeof(PushConstants) == (sizeof(uint32_t) * 4U));

void append_note(std::string& notes, const std::string& note) {
    if (!notes.empty()) {
        notes += ";";
    }
    notes += note;
}

uint64_t compute_output_checksum(const uint32_t* output_values, uint32_t element_count) {
    constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = kOffsetBasis;
    for (uint32_t index = 0U; index < element_count; ++index) {
        hash ^= static_cast<uint64_t>(output_values[index]);
        hash *= kPrime;
    }
    return hash;
}

bool validate_outputs(const uint32_t* output_values, const std::vector<uint32_t>& reference, uint32_t element_count) {
    if (output_values == nullptr || reference.size() < element_count) {
        return false;
    }
    return std::equal(reference.begin(), reference.begin() + element_count, output_values);
}

uint32_t determine_element_count(std::size_t max_buffer_bytes, uint32_t max_dispatch_groups_x,
                                 uint32_t max_image_dimension_2d) {
    // Seven live allocations share the budget: row input, column input, storage
    // image, sampled image, output buffer, and two staging copies, each holding
    // element_count uint32 words (the images are r32ui, one word per texel).
    const uint64_t element_capacity = static_cast<uint64_t>(max_buffer_bytes) / sizeof(uint32_t);
    const uint64_t dispatch_capacity = static_cast<uint64_t>(max_dispatch_groups_x) * kWorkgroupSize;
    // The width is fixed, but the image height grows with the selected budget.
    // Bound both axes before any allocation, including on devices whose image
    // limits are smaller than their legal buffer/dispatch capacity.
    const uint64_t image_capacity = static_cast<uint64_t>(kMaxGridWidth) * max_image_dimension_2d;
    const uint64_t capped = std::min({element_capacity, dispatch_capacity, image_capacity,
                                      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    const uint64_t rounded = capped - (capped % kMaxGridWidth);
    if (rounded < kMaxGridWidth) {
        return 0U;
    }
    return static_cast<uint32_t>(rounded);
}

std::vector<uint32_t> build_problem_sizes(uint32_t max_element_count) {
    const std::array<uint32_t, 6> base_sizes = {
        65536U, 262144U, 1048576U, 4194304U, 8388608U, 16777216U,
    };
    std::vector<uint32_t> sizes;
    sizes.reserve(base_sizes.size() + 1U);
    for (uint32_t candidate : base_sizes) {
        if (candidate <= max_element_count) {
            sizes.push_back(candidate);
        }
    }
    if (sizes.empty() || sizes.back() != max_element_count) {
        sizes.push_back(max_element_count);
    }
    return sizes;
}

// Largest power-of-two width <= kMaxGridWidth that divides element_count.
uint32_t grid_width_for(uint32_t element_count) {
    uint32_t width = 1U;
    while (width < kMaxGridWidth && (element_count % (width * 2U)) == 0U) {
        width *= 2U;
    }
    return width;
}

uint32_t dataset_fetch_value(uint32_t x, uint32_t y, uint32_t grid_width, uint32_t grid_height, RepresentationKind kind,
                             const uint32_t* staging_reference) {
    switch (kind) {
    case RepresentationKind::kRowMajorBuffer:
    case RepresentationKind::kStorageImage:
    case RepresentationKind::kSampledImage:
        return staging_reference[(y * grid_width) + x];
    case RepresentationKind::kColumnMajorBuffer:
        return staging_reference[(x * grid_height) + y];
    }
    return 0U;
}

std::vector<uint32_t> build_reference_outputs(const uint32_t* staged_input_values, uint32_t element_count,
                                              uint32_t grid_width, uint32_t grid_height,
                                              const VariantDescriptor& descriptor) {
    std::vector<uint32_t> reference(element_count, 0U);
    for (uint32_t id = 0U; id < element_count; ++id) {
        uint32_t state = 2166136261U;
        for (uint32_t k = 0U; k < kFetchedTexelsPerInvocation; ++k) {
            uint32_t x = 0U;
            uint32_t y = 0U;
            fetch_coordinates(id, grid_width, grid_height, descriptor.access_mode, k, x, y);
            state = mix_state(state,
                              dataset_fetch_value(x, y, grid_width, grid_height, descriptor.kind, staged_input_values));
        }
        state = mix_state(state, id);
        reference[id] = state;
    }
    return reference;
}

// The staging content at index i is the dataset value of the coordinate that
// layout position i encodes for this representation.
void fill_staging_for_representation(uint32_t* staging_values, uint32_t element_count, uint32_t grid_width,
                                     uint32_t grid_height, RepresentationKind kind) {
    for (uint32_t index = 0U; index < element_count; ++index) {
        uint32_t x = 0U;
        uint32_t y = 0U;
        if (kind == RepresentationKind::kColumnMajorBuffer) {
            x = index / grid_height;
            y = index % grid_height;
        } else {
            x = index % grid_width;
            y = index / grid_width;
        }
        staging_values[index] = dataset_value(x, y);
    }
}

VkDeviceSize compute_span_bytes(uint32_t element_count) {
    return static_cast<VkDeviceSize>(element_count) * static_cast<VkDeviceSize>(sizeof(uint32_t));
}

uint64_t compute_estimated_request_bytes(uint32_t element_count) {
    // Four fetched words plus one written result per invocation, in every variant.
    return static_cast<uint64_t>(element_count) * (kFetchedTexelsPerInvocation + 1U) * sizeof(uint32_t);
}

uint32_t find_memory_type_index(VkPhysicalDevice physical_device, uint32_t memory_type_bits,
                                VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t type_index = 0U; type_index < memory_properties.memoryTypeCount; ++type_index) {
        const uint32_t candidate_bit = 1U << type_index;
        if ((memory_type_bits & candidate_bit) == 0U) {
            continue;
        }
        if ((memory_properties.memoryTypes[type_index].propertyFlags & properties) == properties) {
            return type_index;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool create_image_resource(VkPhysicalDevice physical_device, VkDevice device, uint32_t width, uint32_t height,
                           VkImageUsageFlags usage, VkImageAspectFlagBits aspect, VkImageUsageFlags view_usage,
                           ImageResource& out_resource) {
    // VkSampleCountFlagBits has no zero enumerator, so value-initialization of
    // VkImageCreateInfo is flagged; every field below is assigned explicitly.
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R32_UINT;
    image_info.extent = VkExtent3D{.width = width, .height = height, .depth = 1U};
    image_info.mipLevels = 1U;
    image_info.arrayLayers = 1U;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    const VkResult create_result = vkCreateImage(device, &image_info, nullptr, &out_resource.image);
    if (create_result != VK_SUCCESS) {
        std::cerr << "vkCreateImage failed with error code " << create_result << ".\n";
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, out_resource.image, &requirements);
    const uint32_t type_index =
        find_memory_type_index(physical_device, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type_index == std::numeric_limits<uint32_t>::max()) {
        std::cerr << "No device-local memory type fits the representation image requirements.\n";
        return false;
    }

    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = type_index;
    const VkResult allocate_result = vkAllocateMemory(device, &allocate_info, nullptr, &out_resource.memory);
    if (allocate_result != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory failed for representation image with error code " << allocate_result << ".\n";
        return false;
    }
    const VkResult bind_result = vkBindImageMemory(device, out_resource.image, out_resource.memory, 0U);
    if (bind_result != VK_SUCCESS) {
        std::cerr << "vkBindImageMemory failed with error code " << bind_result << ".\n";
        return false;
    }

    VkImageViewUsageCreateInfo usage_info{};
    usage_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
    usage_info.usage = view_usage;
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.pNext = &usage_info;
    view_info.image = out_resource.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R32_UINT;
    view_info.subresourceRange = VkImageSubresourceRange{.aspectMask = static_cast<VkImageAspectFlags>(aspect),
                                                         .baseMipLevel = 0U,
                                                         .levelCount = 1U,
                                                         .baseArrayLayer = 0U,
                                                         .layerCount = 1U};
    const VkResult view_result = vkCreateImageView(device, &view_info, nullptr, &out_resource.view);
    if (view_result != VK_SUCCESS) {
        std::cerr << "vkCreateImageView failed with error code " << view_result << ".\n";
        return false;
    }

    out_resource.allocation_size = requirements.size;
    out_resource.memory_type_index = type_index;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    out_resource.heap_index = memory_properties.memoryTypes[type_index].heapIndex;
    return true;
}

void destroy_image_resource(VkDevice device, ImageResource& resource) {
    if (resource.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, resource.view, nullptr);
        resource.view = VK_NULL_HANDLE;
    }
    if (resource.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, resource.image, nullptr);
        resource.image = VK_NULL_HANDLE;
    }
    if (resource.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, resource.memory, nullptr);
        resource.memory = VK_NULL_HANDLE;
    }
}

void record_image_layout_transition(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_layout,
                                    VkImageLayout new_layout, VkAccessFlags src_access, VkAccessFlags dst_access,
                                    VkPipelineStageFlagBits src_stage, VkPipelineStageFlagBits dst_stage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                       .baseMipLevel = 0U,
                                                       .levelCount = 1U,
                                                       .baseArrayLayer = 0U,
                                                       .layerCount = 1U};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0U, 0U, nullptr, 0U, nullptr, 1U, &barrier);
}

bool create_buffer_resources(VulkanContext& context, VkDeviceSize span_bytes, BufferResources& out_resources) {
    const VkBufferUsageFlags device_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes, device_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.row_input_buffer)) {
        std::cerr << "Failed to create row-major input buffer.\n";
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes, device_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.column_input_buffer)) {
        std::cerr << "Failed to create column-major input buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.row_input_buffer);
        return false;
    }
    if (!create_buffer_resource(context.physical_device(), context.device(), span_bytes, device_usage,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out_resources.output_buffer)) {
        std::cerr << "Failed to create representation output buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.column_input_buffer);
        destroy_buffer_resource(context.device(), out_resources.row_input_buffer);
        return false;
    }
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.staging_in)) {
        std::cerr << "Failed to create representation upload staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.column_input_buffer);
        destroy_buffer_resource(context.device(), out_resources.row_input_buffer);
        return false;
    }
    if (!create_buffer_resource(
            context.physical_device(), context.device(), span_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out_resources.staging_out)) {
        std::cerr << "Failed to create representation readback staging buffer.\n";
        destroy_buffer_resource(context.device(), out_resources.staging_in);
        destroy_buffer_resource(context.device(), out_resources.output_buffer);
        destroy_buffer_resource(context.device(), out_resources.column_input_buffer);
        destroy_buffer_resource(context.device(), out_resources.row_input_buffer);
        return false;
    }

    if (!map_buffer_memory(context, out_resources.staging_in, "representation upload staging",
                           out_resources.staging_in_mapped_ptr) ||
        !map_buffer_memory(context, out_resources.staging_out, "representation readback staging",
                           out_resources.staging_out_mapped_ptr)) {
        return false;
    }
    return true;
}

void destroy_buffer_resources(VulkanContext& context, BufferResources& resources) {
    if (resources.staging_out_mapped_ptr != nullptr && resources.staging_out.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_out.memory);
        resources.staging_out_mapped_ptr = nullptr;
    }
    if (resources.staging_in_mapped_ptr != nullptr && resources.staging_in.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(context.device(), resources.staging_in.memory);
        resources.staging_in_mapped_ptr = nullptr;
    }
    destroy_buffer_resource(context.device(), resources.staging_out);
    destroy_buffer_resource(context.device(), resources.staging_in);
    destroy_buffer_resource(context.device(), resources.output_buffer);
    destroy_buffer_resource(context.device(), resources.column_input_buffer);
    destroy_buffer_resource(context.device(), resources.row_input_buffer);
}

void buffer_descriptor_bindings(VulkanContext& context, VkDescriptorSet descriptor_set, uint32_t binding,
                                const BufferResource& buffer) {
    const VkDescriptorBufferInfo info{.buffer = buffer.buffer, .offset = 0U, .range = buffer.size};
    VulkanComputeUtils::update_descriptor_set_buffers(
        context.device(), descriptor_set,
        {
            {.binding = binding, .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer_info = info},
        });
}

void image_descriptor_binding(VkDevice device, VkDescriptorSet descriptor_set, uint32_t binding,
                              VkDescriptorType descriptor_type, VkImageView view, VkImageLayout layout,
                              VkSampler sampler) {
    VkDescriptorImageInfo image_info{};
    image_info.sampler = sampler;
    image_info.imageView = view;
    image_info.imageLayout = layout;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = binding;
    write.descriptorCount = 1U;
    write.descriptorType = descriptor_type;
    write.pImageInfo = &image_info;
    vkUpdateDescriptorSets(device, 1U, &write, 0U, nullptr);
}

bool create_pipeline_resources(VulkanContext& context, const std::string& shader_path, RepresentationKind kind,
                               const BufferResources& buffers, const ImageResource& storage_image,
                               const ImageResource& sampled_image, const SamplerResource& sampler,
                               RepresentationPipelineResources& out_resources) {
    if (!VulkanComputeUtils::load_shader_module_from_file(context.device(), shader_path, out_resources.shader_module)) {
        std::cerr << "Failed to load representation shader module: " << shader_path << "\n";
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorPoolSize> pool_sizes;
    if (kind == RepresentationKind::kStorageImage) {
        bindings = {
            {.binding = 0U,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
            {.binding = 1U,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
        };
        pool_sizes = {
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1U},
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1U},
        };
    } else if (kind == RepresentationKind::kSampledImage) {
        bindings = {
            {.binding = 0U,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
            {.binding = 1U,
             .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
        };
        pool_sizes = {
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1U},
            {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1U},
        };
    } else {
        bindings = {
            {.binding = 0U,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
            {.binding = 1U,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1U,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
             .pImmutableSamplers = nullptr},
        };
        pool_sizes = {
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2U},
        };
    }

    if (!VulkanComputeUtils::create_descriptor_set_layout(context.device(), bindings,
                                                          out_resources.descriptor_set_layout)) {
        std::cerr << "Failed to create representation descriptor set layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_descriptor_pool(context.device(), pool_sizes, 1U, out_resources.descriptor_pool)) {
        std::cerr << "Failed to create representation descriptor pool.\n";
        return false;
    }

    if (!VulkanComputeUtils::allocate_descriptor_set(context.device(), out_resources.descriptor_pool,
                                                     out_resources.descriptor_set_layout,
                                                     out_resources.descriptor_set)) {
        std::cerr << "Failed to allocate representation descriptor set.\n";
        return false;
    }

    buffer_descriptor_bindings(context, out_resources.descriptor_set, 0U, buffers.output_buffer);
    switch (kind) {
    case RepresentationKind::kRowMajorBuffer:
        buffer_descriptor_bindings(context, out_resources.descriptor_set, 1U, buffers.row_input_buffer);
        break;
    case RepresentationKind::kColumnMajorBuffer:
        buffer_descriptor_bindings(context, out_resources.descriptor_set, 1U, buffers.column_input_buffer);
        break;
    case RepresentationKind::kStorageImage:
        image_descriptor_binding(context.device(), out_resources.descriptor_set, 1U, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                 storage_image.view, VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE);
        break;
    case RepresentationKind::kSampledImage:
        image_descriptor_binding(context.device(), out_resources.descriptor_set, 1U,
                                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampled_image.view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sampler.sampler);
        break;
    }

    const std::vector<VkPushConstantRange> push_constant_ranges = {
        {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0U, .size = static_cast<uint32_t>(sizeof(PushConstants))},
    };
    if (!VulkanComputeUtils::create_pipeline_layout(context.device(), {out_resources.descriptor_set_layout},
                                                    push_constant_ranges, out_resources.pipeline_layout)) {
        std::cerr << "Failed to create representation pipeline layout.\n";
        return false;
    }

    if (!VulkanComputeUtils::create_compute_pipeline(context.device(), out_resources.shader_module,
                                                     out_resources.pipeline_layout, "main", out_resources.pipeline)) {
        std::cerr << "Failed to create representation compute pipeline.\n";
        return false;
    }
    return true;
}

void destroy_pipeline_resources(VulkanContext& context, RepresentationPipelineResources& resources) {
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

double run_input_upload_stage(VulkanContext& context, const BufferResource& staging, const BufferResource& destination,
                              VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, staging.buffer, destination.buffer, 1U, &copy_region);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer, destination.buffer,
                                                                                bytes);
    });
}

double run_image_transition_stage(VulkanContext& context, VkImage image, VkImageLayout old_layout,
                                  VkImageLayout new_layout, VkAccessFlags src_access, VkAccessFlags dst_access,
                                  VkPipelineStageFlagBits src_stage, VkPipelineStageFlagBits dst_stage) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        record_image_layout_transition(command_buffer, image, old_layout, new_layout, src_access, dst_access, src_stage,
                                       dst_stage);
    });
}

double run_output_poison_stage(VulkanContext& context, const BufferResources& buffers,
                               const std::vector<uint32_t>& reference) {
    auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
    std::ranges::transform(reference, staging_values, [](uint32_t value) {
        return ~value;
    });
    const VkDeviceSize bytes = compute_span_bytes(static_cast<uint32_t>(reference.size()));
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        const VkBufferMemoryBarrier write_barrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                                  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                                  .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                                  .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                  .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                  .buffer = buffers.output_buffer.buffer,
                                                  .offset = 0U,
                                                  .size = bytes};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0U,
                             0U, nullptr, 1U, &write_barrier, 0U, nullptr);
        const VkBufferCopy copy{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.staging_in.buffer, buffers.output_buffer.buffer, 1U, &copy);
        VulkanComputeUtils::record_transfer_write_to_compute_read_write_barrier(command_buffer,
                                                                                buffers.output_buffer.buffer, bytes);
    });
}

double run_image_upload_stage(VulkanContext& context, const BufferResource& staging, VkImage image, uint32_t width,
                              uint32_t height, VkDeviceSize bytes) {
    static_cast<void>(bytes);
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0U;
        region.imageSubresource = VkImageSubresourceLayers{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0U, .baseArrayLayer = 0U, .layerCount = 1U};
        region.imageExtent = VkExtent3D{.width = width, .height = height, .depth = 1U};
        vkCmdCopyBufferToImage(command_buffer, staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U,
                               &region);
    });
}

double run_dispatch_stage(VulkanContext& context, const RepresentationPipelineResources& pipeline_resources,
                          uint32_t element_count, uint32_t grid_width, uint32_t grid_height, uint32_t access_mode,
                          uint32_t max_dispatch_groups_x) {
    const uint32_t group_count_x = VulkanComputeUtils::compute_group_count_1d(element_count, kWorkgroupSize);
    if (group_count_x == 0U || group_count_x > max_dispatch_groups_x) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const PushConstants constants{
        .element_count = element_count,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .access_mode = access_mode,
    };

    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_resources.pipeline_layout, 0U,
                                1U, &pipeline_resources.descriptor_set, 0U, nullptr);
        vkCmdPushConstants(command_buffer, pipeline_resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                           static_cast<uint32_t>(sizeof(constants)), &constants);
        vkCmdDispatch(command_buffer, group_count_x, 1U, 1U);
    });
}

double run_readback_stage(VulkanContext& context, const BufferResources& buffers, VkDeviceSize bytes) {
    return context.measure_gpu_time_ms([&](VkCommandBuffer command_buffer) {
        VulkanComputeUtils::record_compute_write_to_transfer_read_barrier(command_buffer, buffers.output_buffer.buffer,
                                                                          bytes);
        const VkBufferCopy copy_region{.srcOffset = 0U, .dstOffset = 0U, .size = bytes};
        vkCmdCopyBuffer(command_buffer, buffers.output_buffer.buffer, buffers.staging_out.buffer, 1U, &copy_region);

        const VkBufferMemoryBarrier host_read_barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffers.staging_out.buffer,
            .offset = 0U,
            .size = bytes,
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0U, 0U,
                             nullptr, 1U, &host_read_barrier, 0U, nullptr);
    });
}

void record_case_notes(std::string& notes, const VariantDescriptor& descriptor, uint32_t element_count,
                       uint32_t grid_width, uint32_t grid_height, uint64_t request_bytes, uint64_t checksum,
                       uint32_t source_span_bytes, uint64_t scratch_size_bytes, const ImageResource* image_resource,
                       double upload_ms, double image_transition_ms, bool correctness_pass, bool dispatch_ok) {
    append_note(notes, "element_count=" + std::to_string(element_count));
    append_note(notes, "representation=" + representation_name(descriptor.kind));
    append_note(notes,
                "access_mode=" + std::string(descriptor.access_mode == kAccessRandomScatter ? "random_scatter"
                                                                                            : "neighborhood_clamped"));
    append_note(notes, "grid_width=" + std::to_string(grid_width));
    append_note(notes, "grid_height=" + std::to_string(grid_height));
    append_note(notes, "fetches_per_invocation=" + std::to_string(kFetchedTexelsPerInvocation));
    append_note(notes, "local_size_x=" + std::to_string(kWorkgroupSize));
    append_note(notes, "scratch_size_bytes=" + std::to_string(scratch_size_bytes));
    append_note(notes, "live_buffer_count=7");
    append_note(notes, "buffer_span_bytes=" + std::to_string(source_span_bytes));
    append_note(notes, "estimated_request_bytes=" + std::to_string(request_bytes));
    append_note(notes, "gbps_mode=estimated_shader_request_bytes");
    append_note(notes, "staging_upload_ms=" + std::to_string(upload_ms));
    if (image_resource != nullptr) {
        append_note(notes, "format=VK_FORMAT_R32_UINT");
        append_note(notes, "tiling=VK_IMAGE_TILING_OPTIMAL");
        append_note(notes, "allocated_image_bytes=" + std::to_string(image_resource->allocation_size));
        append_note(notes, "image_memory_type_index=" + std::to_string(image_resource->memory_type_index));
        append_note(notes, "image_heap_index=" + std::to_string(image_resource->heap_index));
        append_note(notes, "image_transition_ms=" + std::to_string(image_transition_ms));
        append_note(notes, "image_layout=" + std::string(descriptor.kind == RepresentationKind::kStorageImage
                                                             ? "VK_IMAGE_LAYOUT_GENERAL"
                                                             : "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"));
    }
    append_note(notes, "dataset_content=coordinate_hash");
    append_note(notes, "timing_scope=representation_dispatch");
    append_note(notes, "oracle=reference_complement_before_every_dispatch");
    append_note(notes, "output_preparation=transfer_outside_dispatch_timing");
    append_note(notes, "result_checksum=" + std::to_string(checksum));
    if (!dispatch_ok) {
        append_note(notes, "dispatch_ms_non_finite");
    }
    if (!correctness_pass) {
        append_note(notes, "correctness_mismatch");
    }
}

bool run_case(VulkanContext& context, const BenchmarkRunner& runner,
              const RepresentationPipelineResources& pipeline_resources, const BufferResources& buffers,
              const uint32_t* staged_input_values, uint32_t element_count, uint32_t grid_width, uint32_t grid_height,
              uint32_t max_dispatch_groups_x, const VariantDescriptor& descriptor, const ImageResource* image_resource,
              double upload_ms, double image_transition_ms, uint64_t scratch_size_bytes,
              BufferVsImageRepresentationExperimentOutput& output, bool verbose_progress) {
    const VkDeviceSize bytes = compute_span_bytes(element_count);
    auto* output_values = static_cast<uint32_t*>(buffers.staging_out_mapped_ptr);
    if (output_values == nullptr) {
        std::cerr << "[" << kExperimentId << "] Missing mapped readback pointer.\n";
        output.all_points_correct = false;
        return false;
    }

    const std::vector<uint32_t> reference_outputs =
        build_reference_outputs(staged_input_values, element_count, grid_width, grid_height, descriptor);
    const uint64_t request_bytes = compute_estimated_request_bytes(element_count);

    std::vector<double> dispatch_samples;
    dispatch_samples.reserve(static_cast<std::size_t>(std::max(0, runner.timed_iterations())));

    for (int warmup = 0; warmup < runner.warmup_iterations(); ++warmup) {
        if (!std::isfinite(run_output_poison_stage(context, buffers, reference_outputs))) {
            return false;
        }
        const double dispatch_ms = run_dispatch_stage(context, pipeline_resources, element_count, grid_width,
                                                      grid_height, descriptor.access_mode, max_dispatch_groups_x);
        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] warmup " << (warmup + 1) << "/" << runner.warmup_iterations()
                      << " variant=" << descriptor.variant_name << ", dispatch_ms=" << dispatch_ms << "\n";
        }
    }

    for (int iteration = 0; iteration < runner.timed_iterations(); ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        if (!std::isfinite(run_output_poison_stage(context, buffers, reference_outputs))) {
            return false;
        }
        const double dispatch_ms = run_dispatch_stage(context, pipeline_resources, element_count, grid_width,
                                                      grid_height, descriptor.access_mode, max_dispatch_groups_x);
        const double readback_ms = run_readback_stage(context, buffers, bytes);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> end_to_end_ms = end - start;

        const bool dispatch_ok = std::isfinite(dispatch_ms);
        const bool readback_ok = std::isfinite(readback_ms);
        const bool correctness_pass =
            dispatch_ok && readback_ok && validate_outputs(output_values, reference_outputs, element_count);
        const uint64_t checksum = compute_output_checksum(output_values, element_count);
        output.all_points_correct = output.all_points_correct && correctness_pass;
        dispatch_samples.push_back(dispatch_ms);

        std::string notes;
        record_case_notes(notes, descriptor, element_count, grid_width, grid_height, request_bytes, checksum,
                          static_cast<uint32_t>(bytes), scratch_size_bytes, image_resource, upload_ms,
                          image_transition_ms, correctness_pass, dispatch_ok);

        output.rows.push_back(BenchmarkMeasurementRow{
            .experiment_id = kExperimentId,
            .variant = descriptor.variant_name,
            .problem_size = element_count,
            .dispatch_count = 1U,
            .iteration = iteration,
            .gpu_ms = dispatch_ms,
            .end_to_end_ms = end_to_end_ms.count(),
            .throughput = compute_throughput_elements_per_second(element_count, 1U, dispatch_ms),
            .gbps = compute_effective_gbps_from_bytes(request_bytes, dispatch_ms),
            .correctness_pass = correctness_pass,
            .notes = std::move(notes),
        });

        if (verbose_progress) {
            std::cout << "[" << kExperimentId << "] timed " << (iteration + 1) << "/" << runner.timed_iterations()
                      << " variant=" << descriptor.variant_name << ", element_count=" << element_count
                      << ", dispatch_ms=" << dispatch_ms << ", checksum=" << checksum
                      << ", correctness=" << (correctness_pass ? "pass" : "fail") << "\n";
        }
    }

    output.summary_results.push_back(BenchmarkRunner::summarize_samples(
        std::string(kExperimentId) + "_" + descriptor.variant_name + "_elements_" + std::to_string(element_count),
        dispatch_samples));
    return true;
}

bool check_format_support(VkPhysicalDevice physical_device) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R32_UINT, &properties);
    const VkFormatFeatureFlags required_optimal =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((properties.optimalTilingFeatures & required_optimal) != required_optimal) {
        std::cerr << "R32_UINT optimal-tiling features do not support storage, sampled, and transfer usage.\n";
        return false;
    }
    return true;
}

} // namespace

BufferVsImageRepresentationExperimentOutput
run_buffer_vs_image_representation_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                              const BufferVsImageRepresentationExperimentConfig& config) {
    BufferVsImageRepresentationExperimentOutput output{};
    output.all_points_correct = true;

    if (!context.gpu_timestamps_supported()) {
        std::cerr << "buffer vs image representation experiment requires GPU timestamp support.\n";
        output.all_points_correct = false;
        return output;
    }

    const std::string row_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.row_buffer_shader_path, "48_row_major_buffer.comp.spv");
    const std::string column_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.column_buffer_shader_path, "48_column_major_buffer.comp.spv");
    const std::string storage_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.storage_image_shader_path, "48_storage_image.comp.spv");
    const std::string sampled_shader_path =
        VulkanComputeUtils::resolve_shader_path(config.sampled_image_shader_path, "48_sampled_image.comp.spv");
    if (row_shader_path.empty() || column_shader_path.empty() || storage_shader_path.empty() ||
        sampled_shader_path.empty()) {
        std::cerr << "Could not locate SPIR-V shaders for the representation experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &device_properties);
    if (device_properties.limits.maxImageDimension2D < kMaxGridWidth) {
        std::cerr << "Device supports a smaller than required 2D image dimension.\n";
        output.all_points_correct = false;
        return output;
    }
    if (!check_format_support(context.physical_device())) {
        output.all_points_correct = false;
        return output;
    }

    const uint32_t max_dispatch_groups_x = device_properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_element_count = determine_element_count(config.max_buffer_bytes, max_dispatch_groups_x,
                                                               device_properties.limits.maxImageDimension2D);
    if (max_element_count == 0U) {
        std::cerr << "Scratch buffer too small for buffer vs image representation experiment.\n";
        output.all_points_correct = false;
        return output;
    }

    if (config.verbose_progress) {
        std::cout << "[" << kExperimentId << "] shaders: " << row_shader_path << " | " << column_shader_path << " | "
                  << storage_shader_path << " | " << sampled_shader_path << "\n";
        std::cout << "[" << kExperimentId << "] max_elements=" << max_element_count
                  << ", span_bytes_per_buffer=" << compute_span_bytes(max_element_count)
                  << ", warmup_iterations=" << runner.warmup_iterations()
                  << ", timed_iterations=" << runner.timed_iterations() << "\n";
    }

    BufferResources buffers{};
    if (!create_buffer_resources(context, compute_span_bytes(max_element_count), buffers)) {
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    ImageResource storage_image{};
    ImageResource sampled_image{};
    if (!create_image_resource(context.physical_device(), context.device(), kMaxGridWidth,
                               max_element_count / kMaxGridWidth,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_STORAGE_BIT, storage_image) ||
        !create_image_resource(context.physical_device(), context.device(), kMaxGridWidth,
                               max_element_count / kMaxGridWidth,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_USAGE_SAMPLED_BIT, sampled_image)) {
        destroy_image_resource(context.device(), sampled_image);
        destroy_image_resource(context.device(), storage_image);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    SamplerResource sampler{};
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    const VkResult sampler_result = vkCreateSampler(context.device(), &sampler_info, nullptr, &sampler.sampler);
    if (sampler_result != VK_SUCCESS) {
        std::cerr << "vkCreateSampler failed with error code " << sampler_result << ".\n";
        vkDestroySampler(context.device(), sampler.sampler, nullptr);
        destroy_image_resource(context.device(), sampled_image);
        destroy_image_resource(context.device(), storage_image);
        destroy_buffer_resources(context, buffers);
        output.all_points_correct = false;
        return output;
    }

    const std::vector<VariantDescriptor> variants = build_variant_descriptors();
    const std::vector<uint32_t> problem_sizes = build_problem_sizes(max_element_count);
    std::vector<uint32_t> staged_input_values(max_element_count, 0U);

    for (uint32_t element_count : problem_sizes) {
        const uint32_t grid_width = grid_width_for(element_count);
        const uint32_t grid_height = element_count / grid_width;
        const VkDeviceSize bytes = compute_span_bytes(element_count);
        auto* staging_values = static_cast<uint32_t*>(buffers.staging_in_mapped_ptr);
        if (staging_values == nullptr) {
            std::cerr << "[" << kExperimentId << "] Missing mapped upload pointer.\n";
            output.all_points_correct = false;
            continue;
        }

        struct RepresentationResources {
            RepresentationKind kind;
            std::string shader_path;
            ImageResource* image;
        };
        const std::array<RepresentationResources, 4> representation_specs = {
            {{.kind = RepresentationKind::kRowMajorBuffer, .shader_path = row_shader_path, .image = nullptr},
             {.kind = RepresentationKind::kColumnMajorBuffer, .shader_path = column_shader_path, .image = nullptr},
             {.kind = RepresentationKind::kStorageImage, .shader_path = storage_shader_path, .image = &storage_image},
             {.kind = RepresentationKind::kSampledImage, .shader_path = sampled_shader_path, .image = &sampled_image}},
        };

        for (const RepresentationResources& spec : representation_specs) {
            // Upload once per (size, representation), outside all timed dispatches.
            fill_staging_for_representation(staging_values, element_count, grid_width, grid_height, spec.kind);
            double upload_ms = 0.0;
            double image_transition_ms = 0.0;
            if (spec.kind == RepresentationKind::kStorageImage || spec.kind == RepresentationKind::kSampledImage) {
                const double undefined_to_dst = run_image_transition_stage(
                    context, spec.image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0U,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                const double image_copy_ms =
                    run_image_upload_stage(context, buffers.staging_in, spec.image->image, kMaxGridWidth,
                                           element_count / kMaxGridWidth, bytes);
                const VkImageLayout final_layout = spec.kind == RepresentationKind::kStorageImage
                                                       ? VK_IMAGE_LAYOUT_GENERAL
                                                       : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                const double dst_to_final =
                    run_image_transition_stage(context, spec.image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                               final_layout, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                upload_ms = image_copy_ms;
                image_transition_ms = (std::isfinite(undefined_to_dst) && std::isfinite(dst_to_final))
                                          ? undefined_to_dst + dst_to_final
                                          : std::numeric_limits<double>::quiet_NaN();
                if (!std::isfinite(image_copy_ms) || !std::isfinite(image_transition_ms)) {
                    std::cerr << "[" << kExperimentId << "] Image staging produced non-finite timing values.\n";
                    output.all_points_correct = false;
                }
            } else {
                const BufferResource& destination = spec.kind == RepresentationKind::kRowMajorBuffer
                                                        ? buffers.row_input_buffer
                                                        : buffers.column_input_buffer;
                upload_ms = run_input_upload_stage(context, buffers.staging_in, destination, bytes);
            }

            RepresentationPipelineResources pipeline{};
            if (!create_pipeline_resources(context, spec.shader_path, spec.kind, buffers, storage_image, sampled_image,
                                           sampler, pipeline)) {
                destroy_pipeline_resources(context, pipeline);
                output.all_points_correct = false;
                continue;
            }

            std::vector<uint32_t> logic_input_values(element_count, 0U);
            fill_staging_for_representation(logic_input_values.data(), element_count, grid_width, grid_height,
                                            spec.kind);
            for (const VariantDescriptor& variant : variants) {
                if (variant.kind != spec.kind) {
                    continue;
                }
                if (!run_case(context, runner, pipeline, buffers, logic_input_values.data(), element_count, grid_width,
                              grid_height, max_dispatch_groups_x, variant, spec.image, upload_ms, image_transition_ms,
                              config.scratch_size_bytes, output, config.verbose_progress)) {
                    output.all_points_correct = false;
                }
            }
            destroy_pipeline_resources(context, pipeline);
        }
    }

    vkDestroySampler(context.device(), sampler.sampler, nullptr);
    destroy_image_resource(context.device(), sampled_image);
    destroy_image_resource(context.device(), storage_image);
    destroy_buffer_resources(context, buffers);
    return output;
}
