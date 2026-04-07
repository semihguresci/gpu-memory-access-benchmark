#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace ScratchBufferBudget {

inline VkDeviceSize compute_per_buffer_budget(VkDeviceSize total_budget_bytes, uint32_t buffer_count) {
    if (buffer_count == 0U) {
        return 0U;
    }

    return total_budget_bytes / static_cast<VkDeviceSize>(buffer_count);
}

inline VkDeviceSize compute_scaled_budget(VkDeviceSize total_budget_bytes, uint32_t numerator, uint32_t denominator) {
    if (numerator == 0U || denominator == 0U) {
        return 0U;
    }

    // Split the division into quotient and remainder so large scratch budgets can be
    // scaled without overflowing an intermediate multiply before the final division.
    const auto denominator_bytes = static_cast<VkDeviceSize>(denominator);
    const auto numerator_bytes = static_cast<VkDeviceSize>(numerator);
    const VkDeviceSize quotient = total_budget_bytes / denominator_bytes;
    const VkDeviceSize remainder = total_budget_bytes % denominator_bytes;
    return (quotient * numerator_bytes) + ((remainder * numerator_bytes) / denominator_bytes);
}

} // namespace ScratchBufferBudget
