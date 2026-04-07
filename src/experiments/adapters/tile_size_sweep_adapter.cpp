#include "experiments/experiment_contract.hpp"
#include "experiments/tile_size_sweep_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_tile_size_sweep_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                            const AppOptions& options, ExperimentRunOutput& output) {
    TileSizeSweepExperimentOutput experiment_output = run_tile_size_sweep_experiment(
        context, runner,
        TileSizeSweepExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 2U)),
            .scratch_size_bytes = options.scratch_size_bytes,
            .direct_shader_path = "",
            .tiled_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "tile size sweep experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "tile size sweep experiment reported correctness failures.";
        return false;
    }

    return true;
}
