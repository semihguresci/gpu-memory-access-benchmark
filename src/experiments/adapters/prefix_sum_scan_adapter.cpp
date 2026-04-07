#include "experiments/experiment_contract.hpp"
#include "experiments/prefix_sum_scan_experiment.hpp"
#include "utils/app_options.hpp"
#include <utility>

bool run_prefix_sum_scan_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                            const AppOptions& options, ExperimentRunOutput& output) {
    PrefixSumScanExperimentOutput experiment_output =
        run_prefix_sum_scan_experiment(context, runner,
                                       PrefixSumScanExperimentConfig{
                                           .max_buffer_bytes = options.scratch_size_bytes,
                                           .scan_shader_path = "",
                                           .add_offsets_shader_path = "",
                                           .verbose_progress = options.verbose_progress,
                                       });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "prefix sum scan experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "prefix sum scan experiment reported correctness failures.";
        return false;
    }

    return true;
}
