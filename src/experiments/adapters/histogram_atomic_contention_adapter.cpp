#include "experiments/experiment_contract.hpp"
#include "experiments/histogram_atomic_contention_experiment.hpp"
#include "utils/app_options.hpp"

#include <utility>

bool run_histogram_atomic_contention_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                        const AppOptions& options, ExperimentRunOutput& output) {
    HistogramAtomicContentionExperimentOutput experiment_output =
        run_histogram_atomic_contention_experiment(context, runner,
                                                   HistogramAtomicContentionExperimentConfig{
                                                       .max_buffer_bytes = options.scratch_size_bytes,
                                                       .global_shader_path = "",
                                                       .privatized_shader_path = "",
                                                       .verbose_progress = options.verbose_progress,
                                                   });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "histogram atomic contention experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "histogram atomic contention experiment reported correctness failures.";
        return false;
    }

    return true;
}
