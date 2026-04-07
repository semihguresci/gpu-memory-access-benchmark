#include "experiments/experiment_contract.hpp"
#include "experiments/scalar_type_width_sweep_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scalar_type_width_utils.hpp"

#include <cstddef>
#include <limits>
#include <utility>

namespace {

VkDeviceSize total_resident_scalar_width_bytes(uint32_t elements) {
    return ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kFp32, elements) +
           ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kFp16Storage, elements) +
           ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kU32, elements) +
           ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kU16, elements) +
           ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kU8, elements);
}

std::size_t compute_scalar_width_max_buffer_bytes(VkDeviceSize total_budget_bytes) {
    const VkDeviceSize max_elements_budget = total_budget_bytes / static_cast<VkDeviceSize>(sizeof(uint32_t));
    uint32_t low = 0U;
    uint32_t high =
        static_cast<uint32_t>(std::min<VkDeviceSize>(max_elements_budget, std::numeric_limits<uint32_t>::max()));

    while (low < high) {
        const uint32_t mid = low + ((high - low + 1U) / 2U);
        if (total_resident_scalar_width_bytes(mid) <= total_budget_bytes) {
            low = mid;
        } else {
            high = mid - 1U;
        }
    }

    return static_cast<std::size_t>(
        ScalarTypeWidthUtils::buffer_size_for_variant(ScalarTypeWidthUtils::WidthVariant::kFp32, low));
}

} // namespace

bool run_scalar_type_width_sweep_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const AppOptions& options, ExperimentRunOutput& output) {
    ScalarTypeWidthSweepExperimentOutput experiment_output = run_scalar_type_width_sweep_experiment(
        context, runner,
        ScalarTypeWidthSweepExperimentConfig{
            .max_buffer_bytes = compute_scalar_width_max_buffer_bytes(options.scratch_size_bytes),
            .fp32_shader_path = "",
            .fp16_storage_shader_path = "",
            .u32_shader_path = "",
            .u16_shader_path = "",
            .u8_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "scalar type width sweep experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "scalar type width sweep experiment reported correctness failures.";
        return false;
    }

    return true;
}
