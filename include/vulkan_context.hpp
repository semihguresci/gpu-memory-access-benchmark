#pragma once

#include "utils/gpu_timestamp_timer.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vulkan/vulkan.h>

class VulkanContext {
  public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool initialize(bool enable_validation);
    void shutdown();

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const { return physical_device_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue compute_queue() const { return compute_queue_; }
    [[nodiscard]] uint32_t compute_queue_family_index() const { return compute_queue_family_index_; }
    [[nodiscard]] bool gpu_timestamps_supported() const { return gpu_timestamps_supported_; }

    [[nodiscard]] std::string selected_device_name() const;
    [[nodiscard]] uint32_t selected_device_api_version() const;
    [[nodiscard]] uint32_t selected_device_driver_version() const;
    [[nodiscard]] double measure_gpu_time_ms(const std::function<void(VkCommandBuffer)>& record_commands);

  private:
    bool create_instance(bool enable_validation);
    bool pick_physical_device();
    bool create_logical_device();
    bool create_command_resources();
    bool create_timestamp_query_pool();
    bool setup_debug_messenger();

    static std::optional<uint32_t> find_compute_queue_family(VkPhysicalDevice physical_device);
    [[nodiscard]] static bool check_validation_layer_support();
    [[nodiscard]] static bool check_debug_utils_extension_support();
    void destroy_debug_messenger();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence submit_fence_ = VK_NULL_HANDLE;
    VulkanComputeUtils::GpuTimestampTimer gpu_timestamp_timer_{};
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index_ = 0;
    bool gpu_timestamps_supported_ = false;
    bool validation_enabled_ = false;
};
