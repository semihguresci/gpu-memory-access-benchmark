#include "experiments/experiment_contract.hpp"
#include "experiments/stream_compaction_experiment.hpp"
#include "utils/app_options.hpp"

#include <utility>

bool run_stream_compaction_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner,
                                              const AppOptions& options, ExperimentRunOutput& output) {
    StreamCompactionExperimentOutput experiment_output =
        run_stream_compaction_experiment(context, runner,
                                         StreamCompactionExperimentConfig{
                                             .max_buffer_bytes = options.scratch_size_bytes,
                                             .atomic_append_shader_path = "",
                                             .stage1_shader_path = "",
                                             .stage2_shader_path = "",
                                             .stage3_shader_path = "",
                                             .verbose_progress = options.verbose_progress,
                                         });

    output.summary_results = std::move(experiment_output.summary_results);
    output.rows = std::move(experiment_output.rows);

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "stream compaction experiment produced no summary results.";
        return false;
    }

    if (!experiment_output.all_points_correct) {
        output.success = false;
        output.error_message = "stream compaction experiment reported correctness failures.";
        return false;
    }

    return true;
}
