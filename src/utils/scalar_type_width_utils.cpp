#include "utils/scalar_type_width_utils.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace ScalarTypeWidthUtils {

uint32_t storage_units_for_variant(WidthVariant variant, uint32_t elements) {
    switch (variant) {
    case WidthVariant::kFp16Storage:
    case WidthVariant::kU16:
        return (elements / 2U) + (elements % 2U);
    case WidthVariant::kU8:
        return (elements / 4U) + ((elements % 4U) != 0U ? 1U : 0U);
    case WidthVariant::kFp32:
    case WidthVariant::kU32:
    default:
        return elements;
    }
}

VkDeviceSize buffer_size_for_variant(WidthVariant variant, uint32_t elements) {
    return static_cast<VkDeviceSize>(storage_units_for_variant(variant, elements)) * sizeof(uint32_t);
}

double storage_bytes_per_element(WidthVariant variant) {
    switch (variant) {
    case WidthVariant::kFp16Storage:
    case WidthVariant::kU16:
        return 2.0;
    case WidthVariant::kU8:
        return 1.0;
    case WidthVariant::kFp32:
    case WidthVariant::kU32:
    default:
        return 4.0;
    }
}

float clamp_unit(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

float update_scalar(float value) {
    const float v = clamp_unit(value);
    const float updated = (v * 0.98125F) + 0.004375F + ((v * v) * 0.0003F);
    return clamp_unit(updated);
}

float make_seed_scalar(uint32_t index) {
    uint32_t mixed = (index * 1664525U) + 1013904223U;
    mixed ^= (index * 374761393U);
    return static_cast<float>(mixed & 0x00FFFFFFU) * (1.0F / 16777215.0F);
}

uint16_t float_to_half_bits(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    uint32_t mantissa = bits & 0x007FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0U) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        return static_cast<uint16_t>(sign | 0x7E00U);
    }

    int32_t adjusted_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (adjusted_exponent >= 0x1FU) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (adjusted_exponent <= 0) {
        if (adjusted_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        mantissa |= 0x00800000U;
        const uint32_t shift = static_cast<uint32_t>(1 - adjusted_exponent);
        uint32_t half_mantissa = mantissa >> shift;
        if ((half_mantissa & 0x00001000U) != 0U) {
            half_mantissa += 0x00002000U;
        }
        return static_cast<uint16_t>(sign | (half_mantissa >> 13U));
    }

    if ((mantissa & 0x00001000U) != 0U) {
        mantissa += 0x00002000U;
        if ((mantissa & 0x00800000U) != 0U) {
            mantissa = 0U;
            ++adjusted_exponent;
            if (adjusted_exponent >= 0x1F) {
                return static_cast<uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(adjusted_exponent) << 10U) | (mantissa >> 13U));
}

float half_bits_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000U)) << 16U;
    const uint32_t exponent = (static_cast<uint32_t>(bits) >> 10U) & 0x1FU;
    uint32_t mantissa = static_cast<uint32_t>(bits) & 0x03FFU;

    uint32_t float_bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            float_bits = sign;
        } else {
            int32_t normalized_exponent = 127 - 15 + 1;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --normalized_exponent;
            }
            mantissa &= 0x03FFU;
            float_bits = sign | (static_cast<uint32_t>(normalized_exponent) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        float_bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        const uint32_t normalized_exponent = exponent + static_cast<uint32_t>(127 - 15);
        float_bits = sign | (normalized_exponent << 23U) | (mantissa << 13U);
    }

    return std::bit_cast<float>(float_bits);
}

uint16_t quantize_u16(float value) {
    const float clamped = clamp_unit(value);
    const float scaled = (clamped * 65535.0F) + 0.5F;
    if (scaled <= 0.0F) {
        return 0U;
    }
    if (scaled >= 65535.0F) {
        return 65535U;
    }
    return static_cast<uint16_t>(scaled);
}

float dequantize_u16(uint16_t value) {
    return static_cast<float>(value) * (1.0F / 65535.0F);
}

uint8_t quantize_u8(float value) {
    const float clamped = clamp_unit(value);
    const float scaled = (clamped * 255.0F) + 0.5F;
    if (scaled <= 0.0F) {
        return 0U;
    }
    if (scaled >= 255.0F) {
        return 255U;
    }
    return static_cast<uint8_t>(scaled);
}

float dequantize_u8(uint8_t value) {
    return static_cast<float>(value) * (1.0F / 255.0F);
}

uint16_t read_u16_lane(uint32_t word, uint32_t lane) {
    if (lane == 0U) {
        return static_cast<uint16_t>(word & 0xFFFFU);
    }
    return static_cast<uint16_t>((word >> 16U) & 0xFFFFU);
}

void write_u16_lane(uint32_t& word, uint32_t lane, uint16_t value) {
    const uint32_t lane_mask = (lane == 0U) ? 0x0000FFFFU : 0xFFFF0000U;
    const uint32_t lane_bits = (lane == 0U) ? static_cast<uint32_t>(value) : (static_cast<uint32_t>(value) << 16U);
    word = (word & ~lane_mask) | lane_bits;
}

uint8_t read_u8_lane(uint32_t word, uint32_t lane) {
    const uint32_t shift = lane * 8U;
    return static_cast<uint8_t>((word >> shift) & 0xFFU);
}

void write_u8_lane(uint32_t& word, uint32_t lane, uint8_t value) {
    const uint32_t shift = lane * 8U;
    const uint32_t lane_mask = 0xFFU << shift;
    const uint32_t lane_bits = static_cast<uint32_t>(value) << shift;
    word = (word & ~lane_mask) | lane_bits;
}

float expected_variant_value(WidthVariant variant, uint32_t index) {
    switch (variant) {
    case WidthVariant::kFp16Storage: {
        const float seeded = half_bits_to_float(float_to_half_bits(make_seed_scalar(index)));
        const float updated = update_scalar(seeded);
        return half_bits_to_float(float_to_half_bits(updated));
    }
    case WidthVariant::kU16: {
        const float seeded = dequantize_u16(quantize_u16(make_seed_scalar(index)));
        const float updated = update_scalar(seeded);
        return dequantize_u16(quantize_u16(updated));
    }
    case WidthVariant::kU8: {
        const float seeded = dequantize_u8(quantize_u8(make_seed_scalar(index)));
        const float updated = update_scalar(seeded);
        return dequantize_u8(quantize_u8(updated));
    }
    case WidthVariant::kFp32:
    case WidthVariant::kU32:
    default:
        return update_scalar(make_seed_scalar(index));
    }
}

float validation_tolerance(WidthVariant variant) {
    switch (variant) {
    case WidthVariant::kFp16Storage:
        return 2.0e-3F;
    case WidthVariant::kU16:
        return 2.0e-4F;
    case WidthVariant::kU8:
        return 1.0e-2F;
    case WidthVariant::kFp32:
    case WidthVariant::kU32:
    default:
        return 1.0e-6F;
    }
}

} // namespace ScalarTypeWidthUtils
