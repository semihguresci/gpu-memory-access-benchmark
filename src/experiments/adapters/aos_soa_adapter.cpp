#include "experiments/aos_soa_experiment.hpp"
#include "experiments/experiment_contract.hpp"
#include "utils/app_options.hpp"

#include <cstddef>

bool run_aos_soa_experiment_adapter(VulkanContext& context, const BenchmarkRunner& runner, const AppOptions& options,
                                    ExperimentRunOutput& output) {
    output.summary_results = run_aos_soa_experiment(
        context, runner,
        AosSoaExperimentConfig{.max_buffer_bytes = static_cast<std::size_t>(options.scratch_size_bytes),
                               .aos_shader_path = "",
                               .soa_shader_path = ""});

    if (output.summary_results.empty()) {
        output.success = false;
        output.error_message = "AoS vs SoA experiment produced no summary results.";
        return false;
    }

    return true;
}
