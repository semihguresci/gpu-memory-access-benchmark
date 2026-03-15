#include "support/app_options_test_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

TEST(AppOptionsTests, ParsesAllExperimentsAndNormalizesOutputExtension) {
    const std::vector<std::string> available_experiment_ids = {"01_dispatch_basics", "06_aos_vs_soa"};
    const std::vector<std::string> args = {
        "gpu_memory_layout_experiments",
        "--experiment",
        "all",
        "--iterations",
        "7",
        "--warmup",
        "2",
        "--size",
        "8M",
        "--output",
        "results/run",
    };

    const AppOptions options = TestSupport::parse_app_options(args, available_experiment_ids);

    EXPECT_EQ(options.experiment, "all");
    EXPECT_EQ(options.selected_experiment_ids, available_experiment_ids);
    EXPECT_EQ(options.timed_iterations, 7);
    EXPECT_EQ(options.warmup_iterations, 2);
    EXPECT_FALSE(options.verbose_progress);
    EXPECT_EQ(options.scratch_size_bytes, static_cast<VkDeviceSize>(8U * 1024U * 1024U));
    EXPECT_EQ(options.output_path, "results/run.json");
}

TEST(AppOptionsTests, ParsesSpecificExperimentsWithTrimAndDeduplication) {
    const std::vector<std::string> available_experiment_ids = {"01_dispatch_basics", "06_aos_vs_soa"};
    const std::vector<std::string> args = {
        "gpu_memory_layout_experiments",
        "--experiment",
        " 06_aos_vs_soa, 01_dispatch_basics,06_aos_vs_soa ",
        "--iterations",
        "5",
        "--warmup",
        "1",
        "--size",
        "1024",
        "--output",
        "out.json",
    };

    const AppOptions options = TestSupport::parse_app_options(args, available_experiment_ids);

    const std::vector<std::string> expected_ids = {"06_aos_vs_soa", "01_dispatch_basics"};
    EXPECT_EQ(options.selected_experiment_ids, expected_ids);
    EXPECT_EQ(options.scratch_size_bytes, static_cast<VkDeviceSize>(1024));
    EXPECT_EQ(options.output_path, "out.json");
    EXPECT_FALSE(options.verbose_progress);
}

TEST(AppOptionsTests, ParsesVerboseProgressFlag) {
    const std::vector<std::string> available_experiment_ids = {"01_dispatch_basics"};
    const std::vector<std::string> args = {
        "gpu_memory_layout_experiments",
        "--experiment",
        "01_dispatch_basics",
        "--verbose-progress",
    };

    const AppOptions options = TestSupport::parse_app_options(args, available_experiment_ids);

    EXPECT_TRUE(options.verbose_progress);
}

TEST(AppOptionsTests, UnknownExperimentIdExitsWithCodeTwo) {
    EXPECT_EXIT(([] {
                    const std::vector<std::string> available_experiment_ids = {"01_dispatch_basics"};
                    std::vector<std::string> args = {
                        "gpu_memory_layout_experiments",
                        "--experiment",
                        "unknown_experiment",
                    };
                    static_cast<void>(TestSupport::parse_app_options(args, available_experiment_ids));
                    std::exit(0);
                }()),
                ::testing::ExitedWithCode(2), "Unknown experiment id");
}

TEST(AppOptionsTests, InvalidSizeExitsWithCodeTwo) {
    EXPECT_EXIT(([] {
                    const std::vector<std::string> available_experiment_ids = {"01_dispatch_basics"};
                    std::vector<std::string> args = {
                        "gpu_memory_layout_experiments",
                        "--size",
                        "0",
                    };
                    static_cast<void>(TestSupport::parse_app_options(args, available_experiment_ids));
                    std::exit(0);
                }()),
                ::testing::ExitedWithCode(2), "Invalid --size value");
}
