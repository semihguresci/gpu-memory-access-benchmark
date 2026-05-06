#include "experiments/experiment_contract.hpp"
#include "experiments/persistent_threads_work_queues_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <utility>

bool run_persistent_threads_work_queues_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                           const AppOptions& options, ExperimentRunOutput& output) {
    PersistentThreadsWorkQueuesExperimentOutput experiment_output = run_persistent_threads_work_queues_experiment(
        context, runner,
        PersistentThreadsWorkQueuesExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 3U)),
            .shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "persistent threads work queues experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "persistent threads work queues experiment reported correctness failures.";
        return false;
    }

    return true;
}
