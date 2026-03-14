#include "support/filesystem_test_utils.hpp"
#include "utils/vulkan_compute_utils.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(VulkanComputeUtilsTests, ComputeGroupCountReturnsExpectedCeiling) {
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(0U, 64U), 0U);
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(64U, 0U), 0U);
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(1U, 64U), 1U);
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(64U, 64U), 1U);
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(65U, 64U), 2U);
    EXPECT_EQ(VulkanComputeUtils::compute_group_count_1d(1025U, 256U), 5U);
}

TEST(VulkanComputeUtilsTests, ResolveShaderPathPrefersExplicitExistingPath) {
    TestSupport::ScopedTempDirectory temp_dir("gpu_layout_utils_tests");
    const std::filesystem::path shader_path = temp_dir.path() / "custom_shader.comp.spv";

    std::ofstream output(shader_path, std::ios::binary);
    output << "SPV";
    output.close();

    const std::string resolved = VulkanComputeUtils::resolve_shader_path(shader_path.string(), "ignored.comp.spv");
    EXPECT_EQ(std::filesystem::path(resolved), shader_path);
}

TEST(VulkanComputeUtilsTests, ResolveShaderPathFindsRelativeShaderCandidate) {
    TestSupport::ScopedTempDirectory temp_dir("gpu_layout_utils_tests");
    const std::string shader_name = "unit_test_shader.comp.spv";
    const std::filesystem::path shader_dir = temp_dir.path() / "shaders";
    const std::filesystem::path shader_path = shader_dir / shader_name;
    std::filesystem::create_directories(shader_dir);

    std::ofstream output(shader_path, std::ios::binary);
    output << "SPV";
    output.close();

    const TestSupport::ScopedCurrentPath cwd_guard(temp_dir.path());
    const std::string resolved = VulkanComputeUtils::resolve_shader_path("", shader_name);

    ASSERT_FALSE(resolved.empty());
    EXPECT_TRUE(std::filesystem::exists(resolved));
    EXPECT_TRUE(std::filesystem::equivalent(std::filesystem::path(resolved), shader_path));
}

TEST(VulkanComputeUtilsTests, ReadBinaryFileReadsByteContent) {
    TestSupport::ScopedTempDirectory temp_dir("gpu_layout_utils_tests");
    const std::filesystem::path file_path = temp_dir.path() / "data.bin";
    const std::array<std::uint8_t, 4> bytes = {0x01, 0x02, 0xA0, 0xFF};

    std::ofstream output(file_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();

    std::vector<char> data;
    const bool ok = VulkanComputeUtils::read_binary_file(file_path.string(), data);

    ASSERT_TRUE(ok);
    ASSERT_EQ(data.size(), bytes.size());
    EXPECT_EQ(static_cast<std::uint8_t>(data[0]), bytes[0]);
    EXPECT_EQ(static_cast<std::uint8_t>(data[1]), bytes[1]);
    EXPECT_EQ(static_cast<std::uint8_t>(data[2]), bytes[2]);
    EXPECT_EQ(static_cast<std::uint8_t>(data[3]), bytes[3]);
}

TEST(VulkanComputeUtilsTests, ReadBinaryFileRejectsEmptyFile) {
    TestSupport::ScopedTempDirectory temp_dir("gpu_layout_utils_tests");
    const std::filesystem::path file_path = temp_dir.path() / "empty.bin";

    std::ofstream output(file_path, std::ios::binary);
    output.close();

    std::vector<char> data;
    const bool ok = VulkanComputeUtils::read_binary_file(file_path.string(), data);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(data.empty());
}
