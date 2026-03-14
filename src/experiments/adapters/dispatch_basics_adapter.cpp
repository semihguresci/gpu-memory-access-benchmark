#include "experiments/dispatch_basics_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_dispatch_basics_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                            const AppOptions& options, ExperimentRunOutput& output) {
    DispatchBasicsExperimentOutput experiment_output = run_dispatch_basics_experiment(
        context, runner,
        DispatchBasicsExperimentConfig{.max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
                                       .write_shader_path = "",
                                       .noop_shader_path = "",
                                       .include_noop_variant = true});

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "Dispatch basics experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "Dispatch basics experiment reported correctness failures.";
        return false;
    }

    return true;
}
