#include "experiments/experiment_contract.hpp"
#include "experiments/radix_sort_gpu_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <utility>

bool run_radix_sort_gpu_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                           const AppOptions& options, ExperimentRunOutput& output) {
    RadixSortGpuExperimentOutput experiment_output = run_radix_sort_gpu_experiment(
        context, runner,
        RadixSortGpuExperimentConfig{
            .max_buffer_bytes =
                static_cast<std::size_t>(ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 5U)),
            .count_shader_path = "",
            .scan_shader_path = "",
            .scatter_shader_path = "",
            .verbose_progress = options.verbose_progress,
        });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "radix sort GPU experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "radix sort GPU experiment reported correctness failures.";
        return false;
    }

    return true;
}
