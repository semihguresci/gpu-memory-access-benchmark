#include "vulkan_context.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                              VkDebugUtilsMessageTypeFlagsEXT message_type,
                                              const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                              void* user_data) {
    static_cast<void>(message_type);
    static_cast<void>(user_data);
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan Validation] " << callback_data->pMessage << "\n";
    }

    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT make_debug_messenger_create_info() {
    VkDebugUtilsMessengerCreateInfoEXT debug_info{};
    debug_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = debug_callback;
    return debug_info;
}

} // namespace

VulkanContext::~VulkanContext() {
    shutdown();
}

bool VulkanContext::initialize(bool enable_validation) {
    validation_enabled_ = enable_validation;

    if (!create_instance(enable_validation)) {
        return false;
    }

    if (validation_enabled_) {
        if (!setup_debug_messenger()) {
            return false;
        }
    }

    if (!pick_physical_device()) {
        return false;
    }

    if (!create_logical_device()) {
        return false;
    }

    if (!create_command_resources()) {
        return false;
    }

    if (!create_timestamp_query_pool()) {
        return false;
    }

    return true;
}

void VulkanContext::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);

        gpu_timestamp_timer_.shutdown(device_);

        if (submit_fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, submit_fence_, nullptr);
            submit_fence_ = VK_NULL_HANDLE;
        }

        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
            command_buffer_ = VK_NULL_HANDLE;
        }

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    destroy_debug_messenger();

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    compute_queue_family_index_ = 0;
    gpu_timestamps_supported_ = false;
    validation_enabled_ = false;
}

std::string VulkanContext::selected_device_name() const {
    if (physical_device_ == VK_NULL_HANDLE) {
        return "none";
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    return props.deviceName;
}

uint32_t VulkanContext::selected_device_api_version() const {
    if (physical_device_ == VK_NULL_HANDLE) {
        return 0U;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    return props.apiVersion;
}

uint32_t VulkanContext::selected_device_driver_version() const {
    if (physical_device_ == VK_NULL_HANDLE) {
        return 0U;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    return props.driverVersion;
}

double VulkanContext::measure_gpu_time_ms(const std::function<void(VkCommandBuffer)>& record_commands) {
    if (!gpu_timestamps_supported_ || !gpu_timestamp_timer_.is_ready()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (vkResetFences(device_, 1, &submit_fence_) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (vkResetCommandPool(device_, command_pool_, 0) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(command_buffer_, &begin_info) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    gpu_timestamp_timer_.record_start(command_buffer_);

    record_commands(command_buffer_);

    gpu_timestamp_timer_.record_end(command_buffer_);

    if (vkEndCommandBuffer(command_buffer_) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;

    if (vkQueueSubmit(compute_queue_, 1, &submit_info, submit_fence_) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (vkWaitForFences(device_, 1, &submit_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double out_ms = std::numeric_limits<double>::quiet_NaN();
    if (!gpu_timestamp_timer_.resolve_milliseconds(device_, out_ms)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return out_ms;
}

bool VulkanContext::create_instance(bool enable_validation) {
    if (enable_validation && !check_validation_layer_support()) {
        std::cerr << "Validation layer requested but VK_LAYER_KHRONOS_validation is unavailable.\n";
        return false;
    }

    if (enable_validation && !check_debug_utils_extension_support()) {
        std::cerr << "Validation requested but VK_EXT_debug_utils is unavailable.\n";
        return false;
    }

    const VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                     nullptr,
                                     "gpu_memory_layout_experiments",
                                     VK_MAKE_VERSION(0, 1, 0),
                                     "gpu_memory_layout_experiments",
                                     VK_MAKE_VERSION(0, 1, 0),
                                     VK_API_VERSION_1_2};

    std::vector<const char*> extensions;
    if (enable_validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::vector<const char*> layers;
    if (enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_info = make_debug_messenger_create_info();

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    instance_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instance_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

    if (enable_validation) {
        instance_info.pNext = &debug_info;
    }

    const VkResult result = vkCreateInstance(&instance_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed with error code " << result << "\n";
        return false;
    }

    return true;
}

bool VulkanContext::check_validation_layer_support() {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const auto& layer : available_layers) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            return true;
        }
    }

    return false;
}

bool VulkanContext::check_debug_utils_extension_support() {
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_extensions.data());

    for (const auto& extension : available_extensions) {
        if (std::strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            return true;
        }
    }

    return false;
}

bool VulkanContext::setup_debug_messenger() {
    if (instance_ == VK_NULL_HANDLE) {
        return false;
    }

    auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create_fn == nullptr) {
        return false;
    }

    VkDebugUtilsMessengerCreateInfoEXT debug_info = make_debug_messenger_create_info();
    const VkResult result = create_fn(instance_, &debug_info, nullptr, &debug_messenger_);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateDebugUtilsMessengerEXT failed with error code " << result << "\n";
        return false;
    }

    return true;
}

void VulkanContext::destroy_debug_messenger() {
    if (instance_ == VK_NULL_HANDLE || debug_messenger_ == VK_NULL_HANDLE) {
        return;
    }

    auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_fn != nullptr) {
        destroy_fn(instance_, debug_messenger_, nullptr);
    }

    debug_messenger_ = VK_NULL_HANDLE;
}

std::optional<uint32_t> VulkanContext::find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    if (family_count == 0) {
        return std::nullopt;
    }

    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families.data());

    for (uint32_t index = 0; index < family_count; ++index) {
        if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
            return index;
        }
    }

    return std::nullopt;
}

bool VulkanContext::pick_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        std::cerr << "No Vulkan physical devices found.\n";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        const auto compute_index = find_compute_queue_family(candidate);
        if (compute_index.has_value()) {
            physical_device_ = candidate;
            compute_queue_family_index_ = *compute_index;
            return true;
        }
    }

    std::cerr << "No device with a compute queue family found.\n";
    return false;
}

bool VulkanContext::create_logical_device() {
    const float priority = 1.0F;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, compute_queue_family_index_, 1, &priority};

    const VkPhysicalDeviceFeatures features{};

    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queue_info, 0, nullptr, 0, nullptr, &features};

    const VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed with error code " << result << "\n";
        return false;
    }

    vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);
    return true;
}

bool VulkanContext::create_command_resources() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_queue_family_index_;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "vkCreateCommandPool failed.\n";
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer_) != VK_SUCCESS) {
        std::cerr << "vkAllocateCommandBuffers failed.\n";
        return false;
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(device_, &fence_info, nullptr, &submit_fence_) != VK_SUCCESS) {
        std::cerr << "vkCreateFence failed.\n";
        return false;
    }

    return true;
}

bool VulkanContext::create_timestamp_query_pool() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    const float timestamp_period_ns = props.limits.timestampPeriod;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, families.data());

    gpu_timestamps_supported_ = queue_family_count > compute_queue_family_index_ &&
                                families[compute_queue_family_index_].timestampValidBits > 0;

    if (!gpu_timestamps_supported_) {
        return true;
    }

    gpu_timestamp_timer_.shutdown(device_);

    if (!gpu_timestamp_timer_.initialize(device_, timestamp_period_ns)) {
        std::cerr << "vkCreateQueryPool failed. GPU timestamps disabled.\n";
        gpu_timestamps_supported_ = false;
        return true;
    }

    return true;
}
