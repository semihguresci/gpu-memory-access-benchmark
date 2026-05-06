#include "experiments/async_compute_overlap_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <utility>

bool run_async_compute_overlap_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                  const AppOptions& options, ExperimentRunOutput& output) {
    AsyncComputeOverlapExperimentOutput experiment_output = run_async_compute_overlap_experiment(
        context, runner,
        AsyncComputeOverlapExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 4U)),
            .shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "async compute overlap experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "async compute overlap experiment reported correctness failures.";
        return false;
    }

    return true;
}
