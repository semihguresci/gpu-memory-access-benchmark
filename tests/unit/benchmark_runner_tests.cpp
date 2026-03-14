#include "benchmark_runner.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr double kTolerance = 1.0e-9;

} // namespace

TEST(BenchmarkRunnerTests, SummarizeSamplesComputesExpectedStatistics) {
    const std::vector<double> samples = {1.0, 2.0, 3.0, 4.0};

    const BenchmarkResult result = BenchmarkRunner::summarize_samples("case", samples);

    EXPECT_EQ(result.experiment_name, "case");
    EXPECT_EQ(result.sample_count, 4);
    EXPECT_NEAR(result.average_ms, 2.5, kTolerance);
    EXPECT_NEAR(result.min_ms, 1.0, kTolerance);
    EXPECT_NEAR(result.max_ms, 4.0, kTolerance);
    EXPECT_NEAR(result.median_ms, 2.5, kTolerance);
    EXPECT_NEAR(result.p95_ms, 3.85, kTolerance);
}

TEST(BenchmarkRunnerTests, SummarizeSamplesFiltersNonFiniteValues) {
    const std::vector<double> samples = {
        1.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        3.0,
    };

    const BenchmarkResult result = BenchmarkRunner::summarize_samples("filtered", samples);

    EXPECT_EQ(result.sample_count, 2);
    EXPECT_NEAR(result.average_ms, 2.0, kTolerance);
    EXPECT_NEAR(result.min_ms, 1.0, kTolerance);
    EXPECT_NEAR(result.max_ms, 3.0, kTolerance);
    EXPECT_NEAR(result.median_ms, 2.0, kTolerance);
    EXPECT_NEAR(result.p95_ms, 2.9, kTolerance);
}

TEST(BenchmarkRunnerTests, SummarizeSamplesReturnsNanForNoFiniteSamples) {
    const std::vector<double> samples = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
    };

    const BenchmarkResult result = BenchmarkRunner::summarize_samples("empty", samples);

    EXPECT_EQ(result.sample_count, 0);
    EXPECT_TRUE(std::isnan(result.average_ms));
    EXPECT_TRUE(std::isnan(result.min_ms));
    EXPECT_TRUE(std::isnan(result.max_ms));
    EXPECT_TRUE(std::isnan(result.median_ms));
    EXPECT_TRUE(std::isnan(result.p95_ms));
}

TEST(BenchmarkRunnerTests, RunTimedUsesConfiguredWarmupAndTimedIterations) {
    constexpr int warmup_iterations = 2;
    constexpr int timed_iterations = 3;

    const BenchmarkRunner runner(
        BenchmarkConfig{.warmup_iterations = warmup_iterations, .timed_iterations = timed_iterations});

    int invocation_count = 0;
    const BenchmarkResult result = runner.run_timed("timed", [&]() {
        ++invocation_count;
        return static_cast<double>(invocation_count);
    });

    EXPECT_EQ(runner.warmup_iterations(), warmup_iterations);
    EXPECT_EQ(runner.timed_iterations(), timed_iterations);
    EXPECT_EQ(invocation_count, warmup_iterations + timed_iterations);
    EXPECT_EQ(result.sample_count, timed_iterations);
    EXPECT_NEAR(result.min_ms, 3.0, kTolerance);
    EXPECT_NEAR(result.max_ms, 5.0, kTolerance);
    EXPECT_NEAR(result.average_ms, 4.0, kTolerance);
    EXPECT_NEAR(result.median_ms, 4.0, kTolerance);
    EXPECT_NEAR(result.p95_ms, 4.9, kTolerance);
}
