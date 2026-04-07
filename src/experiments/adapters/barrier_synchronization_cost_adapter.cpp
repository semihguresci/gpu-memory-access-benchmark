#include "experiments/barrier_synchronization_cost_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_barrier_synchronization_cost_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                         const AppOptions& options, ExperimentRunOutput& output) {
    BarrierSynchronizationCostExperimentOutput experiment_output = run_barrier_synchronization_cost_experiment(
        context, runner,
        BarrierSynchronizationCostExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 2U)),
            .scratch_size_bytes = options.scratch_size_bytes,
            .flat_shader_path = "",
            .tiled_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "barrier synchronization cost experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "barrier synchronization cost experiment reported correctness failures.";
        return false;
    }

    return true;
}
