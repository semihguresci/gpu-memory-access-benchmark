#include "utils/gpu_timestamp_timer.hpp"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

TEST(GpuTimestampTimerTests, ComputesWrappedDeltaUsingOnlyValidBits) {
    using VulkanComputeUtils::GpuTimestampTimer;
    EXPECT_EQ(GpuTimestampTimer::timestamp_delta_ticks(250U, 5U, 8U), 11U);
    EXPECT_EQ(GpuTimestampTimer::timestamp_delta_ticks(0xABFAU, 0xCD05U, 8U), 11U);
    EXPECT_EQ(GpuTimestampTimer::timestamp_delta_ticks(10U, 17U, 8U), 7U);
}

TEST(GpuTimestampTimerTests, HandlesFullWidthCountersWithoutUndefinedShift) {
    using VulkanComputeUtils::GpuTimestampTimer;
    EXPECT_EQ(GpuTimestampTimer::timestamp_delta_ticks(std::numeric_limits<uint64_t>::max() - 3U, 2U, 64U), 6U);
    EXPECT_EQ(GpuTimestampTimer::timestamp_delta_ticks(100U, 100U, 64U), 0U);
}

TEST(GpuTimestampTimerTests, RejectsInvalidCounterWidths) {
    using VulkanComputeUtils::GpuTimestampTimer;
    EXPECT_FALSE(GpuTimestampTimer::timestamp_delta_ticks(0U, 1U, 0U).has_value());
    EXPECT_FALSE(GpuTimestampTimer::timestamp_delta_ticks(0U, 1U, 65U).has_value());
}
