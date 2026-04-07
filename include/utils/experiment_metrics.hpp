#pragma once

#include <cmath>
#include <cstdint>

namespace ExperimentMetrics {

inline double compute_throughput_elements_per_second(uint32_t element_count, uint32_t dispatch_count,
                                                     double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    const double elements = static_cast<double>(element_count) * static_cast<double>(dispatch_count);
    return (elements * 1000.0) / dispatch_gpu_ms;
}

inline double compute_effective_gbps_from_bytes(uint64_t total_bytes, double dispatch_gpu_ms) {
    if (!std::isfinite(dispatch_gpu_ms) || dispatch_gpu_ms <= 0.0) {
        return 0.0;
    }

    return static_cast<double>(total_bytes) / (dispatch_gpu_ms * 1.0e6);
}

inline double compute_effective_gbps(uint32_t element_count, uint32_t dispatch_count, uint32_t bytes_per_element,
                                     double dispatch_gpu_ms) {
    const uint64_t total_bytes = static_cast<uint64_t>(element_count) * static_cast<uint64_t>(dispatch_count) *
                                 static_cast<uint64_t>(bytes_per_element);
    return compute_effective_gbps_from_bytes(total_bytes, dispatch_gpu_ms);
}

} // namespace ExperimentMetrics
