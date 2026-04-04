#include "experiments/experiment_contract.hpp"
#include "experiments/read_reuse_cache_locality_experiment.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_read_reuse_cache_locality_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                      const AppOptions& options, ExperimentRunOutput& output) {
    ReadReuseCacheLocalityExperimentOutput experiment_output = run_read_reuse_cache_locality_experiment(
        context, runner,
        ReadReuseCacheLocalityExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "read reuse cache locality experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "read reuse cache locality experiment reported correctness failures.";
        return false;
    }

    return true;
}
