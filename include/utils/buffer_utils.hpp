#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

class VulkanContext;

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    uint32_t memory_type_index = 0;
    uint32_t heap_index = 0;
    VkMemoryPropertyFlags memory_properties = 0;
};

bool create_buffer_resource(VkPhysicalDevice physical_device, VkDevice device, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, BufferResource& out_resource);

bool map_buffer_memory(VulkanContext& context, const BufferResource& buffer, const char* label, void*& mapped_ptr);

void destroy_buffer_resource(VkDevice device, BufferResource& resource);
