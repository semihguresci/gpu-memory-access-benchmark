#include "experiments/experiment_contract.hpp"
#include "experiments/subgroup_reduction_variants_experiment.hpp"
#include "utils/app_options.hpp"

#include <utility>

bool run_subgroup_reduction_variants_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                        const AppOptions& options, ExperimentRunOutput& output) {
    SubgroupReductionVariantsExperimentOutput experiment_output =
        run_subgroup_reduction_variants_experiment(context, runner,
                                                   SubgroupReductionVariantsExperimentConfig{
                                                       .max_buffer_bytes = options.scratch_size_bytes,
                                                       .shared_tree_shader_path = "",
                                                       .subgroup_hybrid_shader_path = "",
                                                       .verbose_progress = options.verbose_progress,
                                                   });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "subgroup reduction variants experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "subgroup reduction variants experiment reported correctness failures.";
        return false;
    }

    return true;
}
