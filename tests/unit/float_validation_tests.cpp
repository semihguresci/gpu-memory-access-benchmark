#include "utils/float_validation.hpp"

#include <gtest/gtest.h>

#include <limits>

TEST(FloatValidationTests, RejectsNonfiniteValuesAndTolerances) {
    constexpr float kTolerance = 1.0e-5F;
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    for (float invalid_value : {kInfinity, -kInfinity, kNan}) {
        EXPECT_FALSE(FloatValidation::nearly_equal(invalid_value, 1.0F, kTolerance));
        EXPECT_FALSE(FloatValidation::nearly_equal(1.0F, invalid_value, kTolerance));
        EXPECT_FALSE(FloatValidation::nearly_equal(invalid_value, invalid_value, kTolerance));
        EXPECT_FALSE(FloatValidation::nearly_equal(1.0F, 1.0F, invalid_value));
    }
    EXPECT_FALSE(FloatValidation::nearly_equal(1.0F, 1.0F, -kTolerance));
}

TEST(FloatValidationTests, KeepsAbsoluteFloorAndRelativeToleranceForFiniteValues) {
    constexpr float kTolerance = 1.0e-5F;
    EXPECT_TRUE(FloatValidation::nearly_equal(0.0F, 0.000001F, kTolerance));
    EXPECT_FALSE(FloatValidation::nearly_equal(0.0F, 0.001F, kTolerance));
    EXPECT_TRUE(FloatValidation::nearly_equal(10000.0F, 10000.01F, kTolerance));
    EXPECT_FALSE(FloatValidation::nearly_equal(10000.0F, 10001.0F, kTolerance));
    EXPECT_TRUE(FloatValidation::nearly_equal(1.0F, 1.0F, 0.0F));
    EXPECT_FALSE(FloatValidation::nearly_equal(1.0F, 2.0F, 0.0F));
}
