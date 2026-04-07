#include "experiments/experiment_contract.hpp"
#include "experiments/parallel_reduction_experiment.hpp"
#include "utils/app_options.hpp"
#include <utility>

bool run_parallel_reduction_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                               const AppOptions& options, ExperimentRunOutput& output) {
    ParallelReductionExperimentOutput experiment_output =
        run_parallel_reduction_experiment(context, runner,
                                          ParallelReductionExperimentConfig{
                                              .max_buffer_bytes = options.scratch_size_bytes,
                                              .global_atomic_shader_path = "",
                                              .shared_tree_shader_path = "",
                                              .verbose_progress = options.verbose_progress,
                                          });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "parallel reduction experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "parallel reduction experiment reported correctness failures.";
        return false;
    }

    return true;
}
