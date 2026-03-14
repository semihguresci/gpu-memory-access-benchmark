#include "vulkan_context.hpp"

#include <gtest/gtest.h>

TEST(VulkanContextIntegrationTests, InitializeAndShutdownSmokeTest) {
    VulkanContext context;
    if (!context.initialize(false)) {
        GTEST_SKIP() << "VulkanContext initialization failed in this environment.";
    }

    EXPECT_NE(context.instance(), VK_NULL_HANDLE);
    EXPECT_NE(context.physical_device(), VK_NULL_HANDLE);
    EXPECT_NE(context.device(), VK_NULL_HANDLE);
    EXPECT_FALSE(context.selected_device_name().empty());

    context.shutdown();

    EXPECT_EQ(context.instance(), VK_NULL_HANDLE);
    EXPECT_EQ(context.physical_device(), VK_NULL_HANDLE);
    EXPECT_EQ(context.device(), VK_NULL_HANDLE);
}
