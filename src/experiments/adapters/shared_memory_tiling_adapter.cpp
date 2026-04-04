#include "experiments/experiment_contract.hpp"
#include "experiments/shared_memory_tiling_experiment.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_shared_memory_tiling_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                 const AppOptions& options, ExperimentRunOutput& output) {
    SharedMemoryTilingExperimentOutput experiment_output = run_shared_memory_tiling_experiment(
        context, runner,
        SharedMemoryTilingExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .direct_shader_path = "",
            .tiled_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "shared memory tiling experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "shared memory tiling experiment reported correctness failures.";
        return false;
    }

    return true;
}
