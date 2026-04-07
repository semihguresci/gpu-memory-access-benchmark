#include "experiments/experiment_contract.hpp"
#include "experiments/two_dimensional_locality_transpose_study_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_two_dimensional_locality_transpose_study_experiment_adapter(VulkanContext& context,
                                                                     const BenchmarkRunner& runner,
                                                                     const AppOptions& options,
                                                                     ExperimentRunOutput& output) {
    TwoDimensionalLocalityTransposeStudyExperimentOutput experiment_output =
        run_two_dimensional_locality_transpose_study_experiment(
            context, runner,
            TwoDimensionalLocalityTransposeStudyExperimentConfig{
                .max_buffer_bytes = static_cast<std::size_t>(
                    ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 2U)),
                .scratch_size_bytes = options.scratch_size_bytes,
                .row_major_copy_shader_path = "",
                .naive_transpose_shader_path = "",
                .tiled_transpose_shader_path = "",
                .tiled_transpose_padded_shader_path = "",
                .verbose_progress = options.verbose_progress,
            });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "2D locality transpose study experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "2D locality transpose study experiment reported correctness failures.";
        return false;
    }

    return true;
}
