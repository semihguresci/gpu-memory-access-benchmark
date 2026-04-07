#include "experiments/experiment_contract.hpp"
#include "experiments/std430_std140_packed_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_std430_std140_packed_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                 const AppOptions& options, ExperimentRunOutput& output) {
    Std430Std140PackedExperimentOutput experiment_output = run_std430_std140_packed_experiment(
        context, runner,
        Std430Std140PackedExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_scaled_budget(options.scratch_size_bytes, 7U, 16U)),
            .std140_shader_path = "",
            .std430_shader_path = "",
            .packed_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "std430/std140/packed experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "std430/std140/packed experiment reported correctness failures.";
        return false;
    }

    return true;
}
