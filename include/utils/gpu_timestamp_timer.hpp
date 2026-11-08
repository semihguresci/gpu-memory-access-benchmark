#pragma once

#include <cstdint>
#include <optional>
#include <vulkan/vulkan.h>

namespace VulkanComputeUtils {

class GpuTimestampTimer {
  public:
    GpuTimestampTimer() = default;

    bool initialize(VkDevice device, float timestamp_period_ns, uint32_t valid_bits);
    void shutdown(VkDevice device);

    void record_start(VkCommandBuffer command_buffer) const;
    void record_end(VkCommandBuffer command_buffer) const;

    bool resolve_milliseconds(VkDevice device, double elapsed_wall_ms, double& out_ms) const;
    [[nodiscard]] static std::optional<uint64_t> timestamp_delta_ticks(uint64_t start, uint64_t end,
                                                                       uint32_t valid_bits);
    [[nodiscard]] bool is_ready() const { return query_pool_ != VK_NULL_HANDLE; }

  private:
    static constexpr uint32_t kQueryCount = 2;

    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ns_ = 0.0F;
    uint32_t valid_bits_ = 0;
};

} // namespace VulkanComputeUtils
