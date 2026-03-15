#include "experiments/aos_soa_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_aos_soa_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner, const AppOptions& options,
                                    ExperimentRunOutput& output) {
    AosSoaExperimentOutput experiment_output = run_aos_soa_experiment(
        context, runner,
        AosSoaExperimentConfig{.max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
                               .aos_shader_path = "",
                               .soa_shader_path = "",
                               .verbose_progress = options.verbose_progress});

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "AoS vs SoA experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "AoS vs SoA experiment reported correctness failures.";
        return false;
    }

    return true;
}
