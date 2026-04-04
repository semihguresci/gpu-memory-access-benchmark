#include "experiments/bandwidth_saturation_sweep_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_bandwidth_saturation_sweep_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                       const AppOptions& options, ExperimentRunOutput& output) {
    BandwidthSaturationSweepExperimentOutput experiment_output = run_bandwidth_saturation_sweep_experiment(
        context, runner,
        BandwidthSaturationSweepExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .read_only_shader_path = "",
            .write_only_shader_path = "",
            .read_write_copy_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "Bandwidth saturation sweep experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "Bandwidth saturation sweep experiment reported correctness failures.";
        return false;
    }

    return true;
}
