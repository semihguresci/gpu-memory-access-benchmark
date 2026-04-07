#include "utils/scratch_buffer_budget.hpp"

#include <gtest/gtest.h>

TEST(ScratchBufferBudgetTests, ComputesPerBufferBudgetFromTotalBudget) {
    EXPECT_EQ(ScratchBufferBudget::compute_per_buffer_budget(12U, 3U), 4U);
    EXPECT_EQ(ScratchBufferBudget::compute_per_buffer_budget(13U, 3U), 4U);
    EXPECT_EQ(ScratchBufferBudget::compute_per_buffer_budget(0U, 3U), 0U);
}

TEST(ScratchBufferBudgetTests, ReturnsZeroWhenBufferCountIsZero) {
    EXPECT_EQ(ScratchBufferBudget::compute_per_buffer_budget(4096U, 0U), 0U);
}

TEST(ScratchBufferBudgetTests, ComputesScaledBudgetWithoutOverflowingIntermediateMath) {
    EXPECT_EQ(ScratchBufferBudget::compute_scaled_budget(16U, 7U, 16U), 7U);
    EXPECT_EQ(ScratchBufferBudget::compute_scaled_budget(39U, 16U, 39U), 16U);
    EXPECT_EQ(ScratchBufferBudget::compute_scaled_budget(1300U, 4U, 13U), 400U);
}

TEST(ScratchBufferBudgetTests, ReturnsZeroWhenScaledBudgetInputsAreInvalid) {
    EXPECT_EQ(ScratchBufferBudget::compute_scaled_budget(4096U, 0U, 3U), 0U);
    EXPECT_EQ(ScratchBufferBudget::compute_scaled_budget(4096U, 3U, 0U), 0U);
}
