#include "utils/gpu_timestamp_timer.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace VulkanComputeUtils {

bool GpuTimestampTimer::initialize(VkDevice device, float timestamp_period_ns, uint32_t valid_bits) {
    if (query_pool_ != VK_NULL_HANDLE || !std::isfinite(timestamp_period_ns) || timestamp_period_ns <= 0.0F ||
        valid_bits == 0U || valid_bits > 64U) {
        std::cerr << "Invalid timestamp timer initialization (period, valid bits, or existing query pool).\n";
        return false;
    }
    timestamp_period_ns_ = timestamp_period_ns;
    valid_bits_ = valid_bits;

    VkQueryPoolCreateInfo query_info{};
    query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = kQueryCount;

    return vkCreateQueryPool(device, &query_info, nullptr, &query_pool_) == VK_SUCCESS;
}

void GpuTimestampTimer::shutdown(VkDevice device) {
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
    timestamp_period_ns_ = 0.0F;
    valid_bits_ = 0U;
}

void GpuTimestampTimer::record_start(VkCommandBuffer command_buffer) const {
    vkCmdResetQueryPool(command_buffer, query_pool_, 0, kQueryCount);
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);
}

void GpuTimestampTimer::record_end(VkCommandBuffer command_buffer) const {
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
}

std::optional<uint64_t> GpuTimestampTimer::timestamp_delta_ticks(uint64_t start, uint64_t end, uint32_t valid_bits) {
    if (valid_bits == 0U || valid_bits > 64U) {
        return std::nullopt;
    }
    const uint64_t mask = valid_bits == 64U ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << valid_bits) - 1U;
    return (end - start) & mask;
}

bool GpuTimestampTimer::resolve_milliseconds(VkDevice device, double elapsed_wall_ms, double& out_ms) const {
    if (query_pool_ == VK_NULL_HANDLE || timestamp_period_ns_ <= 0.0F) {
        return false;
    }
    // Host submit-to-fence duration bounds the timestamp interval. Reject a
    // possible complete counter wrap: modulo subtraction cannot recover it.
    const double wrap_ms = std::ldexp(1.0, static_cast<int>(valid_bits_)) * timestamp_period_ns_ / 1.0e6;
    if (!std::isfinite(elapsed_wall_ms) || elapsed_wall_ms < 0.0 || elapsed_wall_ms >= wrap_ms) {
        std::cerr << "Timestamp interval may span a complete counter wrap. Use a shorter measured workload.\n";
        return false;
    }

    std::array<uint64_t, kQueryCount> timestamps{};
    const VkResult result =
        vkGetQueryPoolResults(device, query_pool_, 0, static_cast<uint32_t>(timestamps.size()), sizeof(timestamps),
                              timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS) {
        return false;
    }

    const auto delta = timestamp_delta_ticks(timestamps[0], timestamps[1], valid_bits_);
    if (!delta.has_value()) {
        return false;
    }
    const double ns = static_cast<double>(*delta) * static_cast<double>(timestamp_period_ns_);
    out_ms = ns / 1000000.0;

    return !std::isnan(out_ms) && std::isfinite(out_ms);
}

} // namespace VulkanComputeUtils
