#include "utils/experiment_metrics.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace {

constexpr double kTolerance = 1.0e-9;

} // namespace

TEST(ExperimentMetricsTests, ComputeThroughputElementsPerSecondReturnsZeroForInvalidGpuTime) {
    EXPECT_DOUBLE_EQ(ExperimentMetrics::compute_throughput_elements_per_second(1024U, 4U, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(ExperimentMetrics::compute_throughput_elements_per_second(1024U, 4U, -1.0), 0.0);
}

TEST(ExperimentMetricsTests, ComputeEffectiveGbpsFromBytesReturnsZeroForInvalidInputs) {
    EXPECT_DOUBLE_EQ(ExperimentMetrics::compute_effective_gbps_from_bytes(0U, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(ExperimentMetrics::compute_effective_gbps_from_bytes(4096U, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(ExperimentMetrics::compute_effective_gbps_from_bytes(4096U, -1.0), 0.0);
}

TEST(ExperimentMetricsTests, ComputeEffectiveGbpsMultipliesElementsDispatchesAndBytesPerElement) {
    const double gbps = ExperimentMetrics::compute_effective_gbps(1024U, 4U, 8U, 2.0);

    EXPECT_NEAR(gbps, 0.016384, kTolerance);
}
