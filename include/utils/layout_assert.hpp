#pragma once

#include <cstddef>
#include <type_traits>

#define GPU_LAYOUT_ASSERT_STANDARD_LAYOUT(Type)                                                                        \
    static_assert(std::is_standard_layout_v<Type>, #Type " must be standard-layout")

#define GPU_LAYOUT_ASSERT_TRIVIALLY_COPYABLE(Type)                                                                     \
    static_assert(std::is_trivially_copyable_v<Type>, #Type " must be trivially copyable")

#define GPU_LAYOUT_ASSERT_SIZE(Type, ExpectedSize)                                                                     \
    static_assert(sizeof(Type) == static_cast<std::size_t>(ExpectedSize), #Type " size mismatch")

#define GPU_LAYOUT_ASSERT_ALIGNMENT(Type, ExpectedAlignment)                                                           \
    static_assert(alignof(Type) == static_cast<std::size_t>(ExpectedAlignment), #Type " alignment mismatch")

#define GPU_LAYOUT_ASSERT_OFFSET(Type, Member, ExpectedOffset)                                                         \
    static_assert(offsetof(Type, Member) == static_cast<std::size_t>(ExpectedOffset),                                  \
                  #Type "." #Member " offset mismatch")
