#include "utils/scalar_type_width_utils.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {

using ScalarTypeWidthUtils::WidthVariant;

constexpr float kFloatTolerance = 1.0e-6F;

} // namespace

TEST(ScalarTypeWidthUtilsTests, StorageUnitsForVariantRoundsPackedVariants) {
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kFp32, 0U), 0U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU32, 5U), 5U);

    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kFp16Storage, 1U), 1U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kFp16Storage, 2U), 1U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kFp16Storage, 3U), 2U);

    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU16, 5U), 3U);

    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU8, 1U), 1U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU8, 4U), 1U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU8, 5U), 2U);
}

TEST(ScalarTypeWidthUtilsTests, BufferSizeForVariantUsesPackedStorageUnits) {
    EXPECT_EQ(ScalarTypeWidthUtils::buffer_size_for_variant(WidthVariant::kFp32, 8U),
              static_cast<VkDeviceSize>(8U * sizeof(uint32_t)));
    EXPECT_EQ(ScalarTypeWidthUtils::buffer_size_for_variant(WidthVariant::kFp16Storage, 5U),
              static_cast<VkDeviceSize>(3U * sizeof(uint32_t)));
    EXPECT_EQ(ScalarTypeWidthUtils::buffer_size_for_variant(WidthVariant::kU8, 17U),
              static_cast<VkDeviceSize>(5U * sizeof(uint32_t)));
}

TEST(ScalarTypeWidthUtilsTests, PackedCountsDoNotOverflowAtUint32Limit) {
    constexpr auto kCount = std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kFp16Storage, kCount), 2147483648U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU16, kCount), 2147483648U);
    EXPECT_EQ(ScalarTypeWidthUtils::storage_units_for_variant(WidthVariant::kU8, kCount), 1073741824U);
    EXPECT_EQ(ScalarTypeWidthUtils::buffer_size_for_variant(WidthVariant::kU8, kCount), VkDeviceSize{4294967296ULL});
}

TEST(ScalarTypeWidthUtilsTests, StorageBytesPerElementMatchesVariantWidth) {
    EXPECT_DOUBLE_EQ(ScalarTypeWidthUtils::storage_bytes_per_element(WidthVariant::kFp32), 4.0);
    EXPECT_DOUBLE_EQ(ScalarTypeWidthUtils::storage_bytes_per_element(WidthVariant::kU32), 4.0);
    EXPECT_DOUBLE_EQ(ScalarTypeWidthUtils::storage_bytes_per_element(WidthVariant::kFp16Storage), 2.0);
    EXPECT_DOUBLE_EQ(ScalarTypeWidthUtils::storage_bytes_per_element(WidthVariant::kU16), 2.0);
    EXPECT_DOUBLE_EQ(ScalarTypeWidthUtils::storage_bytes_per_element(WidthVariant::kU8), 1.0);
}

TEST(ScalarTypeWidthUtilsTests, ClampUnitClampsOutOfRangeValues) {
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::clamp_unit(-5.0F), 0.0F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::clamp_unit(0.25F), 0.25F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::clamp_unit(4.0F), 1.0F);
}

TEST(ScalarTypeWidthUtilsTests, UpdateScalarUsesClampedInputAndStaysInUnitRange) {
    EXPECT_NEAR(ScalarTypeWidthUtils::update_scalar(-1.0F), 0.004375F, kFloatTolerance);
    EXPECT_NEAR(ScalarTypeWidthUtils::update_scalar(0.0F), 0.004375F, kFloatTolerance);
    EXPECT_NEAR(ScalarTypeWidthUtils::update_scalar(0.5F), 0.495075F, kFloatTolerance);
    EXPECT_NEAR(ScalarTypeWidthUtils::update_scalar(1.0F), 0.985925F, kFloatTolerance);
    EXPECT_NEAR(ScalarTypeWidthUtils::update_scalar(2.0F), 0.985925F, kFloatTolerance);
}

TEST(ScalarTypeWidthUtilsTests, MakeSeedScalarIsDeterministicAndBounded) {
    const float sample_a = ScalarTypeWidthUtils::make_seed_scalar(42U);
    const float sample_b = ScalarTypeWidthUtils::make_seed_scalar(42U);
    const float sample_c = ScalarTypeWidthUtils::make_seed_scalar(43U);

    EXPECT_FLOAT_EQ(sample_a, sample_b);
    EXPECT_NE(sample_a, sample_c);

    for (uint32_t index = 0U; index < 256U; ++index) {
        const float value = ScalarTypeWidthUtils::make_seed_scalar(index);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, 0.0F);
        EXPECT_LE(value, 1.0F);
    }
}

TEST(ScalarTypeWidthUtilsTests, HalfConversionHandlesCommonValues) {
    EXPECT_EQ(ScalarTypeWidthUtils::float_to_half_bits(1.0F), static_cast<uint16_t>(0x3C00U));
    EXPECT_EQ(ScalarTypeWidthUtils::float_to_half_bits(-2.0F), static_cast<uint16_t>(0xC000U));
    EXPECT_EQ(ScalarTypeWidthUtils::float_to_half_bits(std::numeric_limits<float>::infinity()),
              static_cast<uint16_t>(0x7C00U));
    EXPECT_EQ(ScalarTypeWidthUtils::float_to_half_bits(-std::numeric_limits<float>::infinity()),
              static_cast<uint16_t>(0xFC00U));

    const float round_trip_nan = ScalarTypeWidthUtils::half_bits_to_float(
        ScalarTypeWidthUtils::float_to_half_bits(std::numeric_limits<float>::quiet_NaN()));
    EXPECT_TRUE(std::isnan(round_trip_nan));

    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::half_bits_to_float(0x3C00U), 1.0F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::half_bits_to_float(0xC000U), -2.0F);
    EXPECT_TRUE(std::isinf(ScalarTypeWidthUtils::half_bits_to_float(0x7C00U)));
    EXPECT_TRUE(std::isinf(ScalarTypeWidthUtils::half_bits_to_float(0xFC00U)));
}

TEST(ScalarTypeWidthUtilsTests, QuantizeU16AndDequantizeU16ClampAndRound) {
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u16(-0.1F), 0U);
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u16(0.0F), 0U);
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u16(0.5F), static_cast<uint16_t>(32768U));
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u16(1.0F), static_cast<uint16_t>(65535U));
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u16(1.1F), static_cast<uint16_t>(65535U));

    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::dequantize_u16(0U), 0.0F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::dequantize_u16(65535U), 1.0F);

    const float value = 0.1234F;
    const float round_trip = ScalarTypeWidthUtils::dequantize_u16(ScalarTypeWidthUtils::quantize_u16(value));
    EXPECT_LE(std::fabs(value - round_trip), 1.0F / 65535.0F);
}

TEST(ScalarTypeWidthUtilsTests, QuantizeU8AndDequantizeU8ClampAndRound) {
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u8(-0.1F), 0U);
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u8(0.0F), 0U);
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u8(0.5F), static_cast<uint8_t>(128U));
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u8(1.0F), static_cast<uint8_t>(255U));
    EXPECT_EQ(ScalarTypeWidthUtils::quantize_u8(1.1F), static_cast<uint8_t>(255U));

    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::dequantize_u8(0U), 0.0F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::dequantize_u8(255U), 1.0F);

    const float value = 0.456F;
    const float round_trip = ScalarTypeWidthUtils::dequantize_u8(ScalarTypeWidthUtils::quantize_u8(value));
    EXPECT_LE(std::fabs(value - round_trip), 1.0F / 255.0F);
}

TEST(ScalarTypeWidthUtilsTests, U16LaneReadWritePreservesOppositeLaneBits) {
    uint32_t word = 0U;

    ScalarTypeWidthUtils::write_u16_lane(word, 0U, static_cast<uint16_t>(0x1234U));
    EXPECT_EQ(word, 0x00001234U);
    ScalarTypeWidthUtils::write_u16_lane(word, 1U, static_cast<uint16_t>(0xABCDU));
    EXPECT_EQ(word, 0xABCD1234U);

    EXPECT_EQ(ScalarTypeWidthUtils::read_u16_lane(word, 0U), static_cast<uint16_t>(0x1234U));
    EXPECT_EQ(ScalarTypeWidthUtils::read_u16_lane(word, 1U), static_cast<uint16_t>(0xABCDU));

    ScalarTypeWidthUtils::write_u16_lane(word, 0U, static_cast<uint16_t>(0x0F0FU));
    EXPECT_EQ(word, 0xABCD0F0FU);
    EXPECT_EQ(ScalarTypeWidthUtils::read_u16_lane(word, 1U), static_cast<uint16_t>(0xABCDU));
}

TEST(ScalarTypeWidthUtilsTests, U8LaneReadWritePreservesOtherLaneBits) {
    uint32_t word = 0U;

    ScalarTypeWidthUtils::write_u8_lane(word, 0U, static_cast<uint8_t>(0x11U));
    ScalarTypeWidthUtils::write_u8_lane(word, 1U, static_cast<uint8_t>(0x22U));
    ScalarTypeWidthUtils::write_u8_lane(word, 2U, static_cast<uint8_t>(0x33U));
    ScalarTypeWidthUtils::write_u8_lane(word, 3U, static_cast<uint8_t>(0x44U));

    EXPECT_EQ(word, 0x44332211U);
    EXPECT_EQ(ScalarTypeWidthUtils::read_u8_lane(word, 0U), static_cast<uint8_t>(0x11U));
    EXPECT_EQ(ScalarTypeWidthUtils::read_u8_lane(word, 1U), static_cast<uint8_t>(0x22U));
    EXPECT_EQ(ScalarTypeWidthUtils::read_u8_lane(word, 2U), static_cast<uint8_t>(0x33U));
    EXPECT_EQ(ScalarTypeWidthUtils::read_u8_lane(word, 3U), static_cast<uint8_t>(0x44U));
}

TEST(ScalarTypeWidthUtilsTests, ExpectedVariantValueRespectsToleranceEnvelope) {
    constexpr std::array<uint32_t, 6> kIndices = {0U, 1U, 17U, 255U, 1024U, 4095U};

    for (const uint32_t index : kIndices) {
        const float fp32 = ScalarTypeWidthUtils::expected_variant_value(WidthVariant::kFp32, index);
        const float u32 = ScalarTypeWidthUtils::expected_variant_value(WidthVariant::kU32, index);
        const float fp16 = ScalarTypeWidthUtils::expected_variant_value(WidthVariant::kFp16Storage, index);
        const float u16 = ScalarTypeWidthUtils::expected_variant_value(WidthVariant::kU16, index);
        const float u8 = ScalarTypeWidthUtils::expected_variant_value(WidthVariant::kU8, index);

        EXPECT_FLOAT_EQ(fp32, u32);

        EXPECT_TRUE(std::isfinite(fp32));
        EXPECT_TRUE(std::isfinite(fp16));
        EXPECT_TRUE(std::isfinite(u16));
        EXPECT_TRUE(std::isfinite(u8));

        EXPECT_GE(fp32, 0.0F);
        EXPECT_LE(fp32, 1.0F);
        EXPECT_GE(fp16, 0.0F);
        EXPECT_LE(fp16, 1.0F);
        EXPECT_GE(u16, 0.0F);
        EXPECT_LE(u16, 1.0F);
        EXPECT_GE(u8, 0.0F);
        EXPECT_LE(u8, 1.0F);

        EXPECT_LE(std::fabs(fp16 - fp32), ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kFp16Storage));
        EXPECT_LE(std::fabs(u16 - fp32), ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kU16));
        EXPECT_LE(std::fabs(u8 - fp32), ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kU8));
    }
}

TEST(ScalarTypeWidthUtilsTests, ValidationToleranceMatchesVariantPolicy) {
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kFp32), 1.0e-6F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kU32), 1.0e-6F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kFp16Storage), 2.0e-3F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kU16), 2.0e-4F);
    EXPECT_FLOAT_EQ(ScalarTypeWidthUtils::validation_tolerance(WidthVariant::kU8), 1.0e-2F);
}
