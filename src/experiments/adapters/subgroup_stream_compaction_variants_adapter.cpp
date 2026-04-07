#include "experiments/experiment_contract.hpp"
#include "experiments/subgroup_stream_compaction_variants_experiment.hpp"
#include "utils/app_options.hpp"
#include "utils/scratch_buffer_budget.hpp"

#include <cstddef>
#include <utility>

bool run_subgroup_stream_compaction_variants_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                                                const AppOptions& options,
                                                                ExperimentRunOutput& output) {
    SubgroupStreamCompactionVariantsExperimentOutput experiment_output =
        run_subgroup_stream_compaction_variants_experiment(
            context, runner,
            SubgroupStreamCompactionVariantsExperimentConfig{
                .max_buffer_bytes = static_cast<std::size_t>(
                    ScratchBufferBudget::compute_per_buffer_budget(options.scratch_size_bytes, 3U)),
                .shared_atomic_shader_path = "",
                .subgroup_ballot_shader_path = "",
                .verbose_progress = options.verbose_progress,
            });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "subgroup stream compaction variants experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "subgroup stream compaction variants experiment reported correctness failures.";
        return false;
    }

    return true;
}
