#include "experiments/experiment_contract.hpp"
#include "experiments/occupancy_modeling_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <utility>

bool run_occupancy_modeling_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                               const AppOptions& options, ExperimentRunOutput& output) {
    OccupancyModelingExperimentOutput experiment_output = run_occupancy_modeling_experiment(
        context, runner,
        OccupancyModelingExperimentConfig{
            .max_buffer_bytes =
                static_cast<std::size_t>(ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 2U)),
            .low_smem_shader_path = "",
            .medium_smem_shader_path = "",
            .high_smem_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "occupancy modeling experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "occupancy modeling experiment reported correctness failures.";
        return false;
    }

    return true;
}
