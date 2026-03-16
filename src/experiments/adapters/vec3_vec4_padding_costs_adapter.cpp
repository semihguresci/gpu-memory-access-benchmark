#include "experiments/experiment_contract.hpp"
#include "experiments/vec3_vec4_padding_costs_experiment.hpp"
#include "utils/app_options.hpp"

#include <cstddef>
#include <utility>

bool run_vec3_vec4_padding_costs_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                    const AppOptions& options, ExperimentRunOutput& output) {
    Vec3Vec4PaddingCostsExperimentOutput experiment_output = run_vec3_vec4_padding_costs_experiment(
        context, runner,
        Vec3Vec4PaddingCostsExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .vec3_shader_path = "",
            .vec4_shader_path = "",
            .split_scalars_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "vec3/vec4 padding cost experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "vec3/vec4 padding cost experiment reported correctness failures.";
        return false;
    }

    return true;
}

