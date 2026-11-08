#include "experiments/buffer_vs_image_representation_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_buffer_vs_image_representation_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                           const AppOptions& options, ExperimentRunOutput& output) {
    BufferVsImageRepresentationExperimentOutput experiment_output = run_buffer_vs_image_representation_experiment(
        context, runner,
        BufferVsImageRepresentationExperimentConfig{
            .max_buffer_bytes = static_cast<std::size_t>(
                ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 7U)),
            .scratch_size_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
            .row_buffer_shader_path = "",
            .column_buffer_shader_path = "",
            .storage_image_shader_path = "",
            .sampled_image_shader_path = "",
            .verbose_progress = options.verbose_progress});

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "Buffer vs image representation experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "Buffer vs image representation experiment reported correctness failures.";
        return false;
    }

    return true;
}
