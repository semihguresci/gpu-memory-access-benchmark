#include "experiments/dependent_load_latency_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_dependent_load_latency_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                   const AppOptions& options, ExperimentRunOutput& output) {
    DependentLoadLatencyExperimentOutput experiment_output = run_dependent_load_latency_experiment(
        context, runner,
        DependentLoadLatencyExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 4U)),
            .scratch_size_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .shader_path = "",
            .verbose_progress = options.verbose_progress});

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "Dependent load latency experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "Dependent load latency experiment reported correctness failures.";
        return false;
    }

    return true;
}
