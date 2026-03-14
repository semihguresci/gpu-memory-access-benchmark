#include "benchmark_runner.hpp"
#include "experiments/experiment_registry.hpp"
#include "utils/app_options.hpp"
#include "utils/json_exporter.hpp"
#include "vulkan_context.hpp"

#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

std::string format_vulkan_version(uint32_t version) {
    if (version == 0U) {
        return {};
    }

    std::ostringstream stream;
    stream << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version);
    return stream.str();
}

} // namespace

int main(int argc, char** argv) noexcept {
    try {
        const std::vector<std::string> available_experiment_ids = enabled_experiment_ids();
        const AppOptions options = ArgumentParser::parse(argc, argv, available_experiment_ids);

        VulkanContext context;
        if (!context.initialize(options.enable_validation)) {
            std::cerr << "Vulkan initialization failed.\n";
            return 1;
        }

        std::cout << "Using GPU: " << context.selected_device_name() << "\n";
        std::cout << "Validation: " << (options.enable_validation ? "enabled" : "disabled") << "\n";
        std::cout << "GPU timestamps: " << (context.gpu_timestamps_supported() ? "supported" : "not supported") << "\n";
        if (!context.gpu_timestamps_supported()) {
            std::cerr << "GPU timestamp queries are required for this benchmark run.\n";
            context.shutdown();
            return 1;
        }

        BenchmarkRunner runner(
            {.warmup_iterations = options.warmup_iterations, .timed_iterations = options.timed_iterations});
        std::vector<BenchmarkResult> results;
        std::vector<BenchmarkMeasurementRow> rows;

        for (const std::string& experiment_id : options.selected_experiment_ids) {
            const ExperimentDescriptor* descriptor = find_experiment_descriptor(experiment_id);
            if (descriptor == nullptr) {
                std::cerr << "Unknown experiment id selected after parsing: " << experiment_id << "\n";
                context.shutdown();
                return 1;
            }

            if (!descriptor->enabled) {
                std::cerr << "Experiment is currently disabled: " << descriptor->id << "\n";
                context.shutdown();
                return 1;
            }

            if (descriptor->run == nullptr) {
                std::cerr << "Experiment has no run function: " << descriptor->id << "\n";
                context.shutdown();
                return 1;
            }

            ExperimentRunOutput run_output{};
            const bool run_ok = descriptor->run(context, runner, options, run_output);
            if (!run_ok || !run_output.success) {
                std::cerr << "Experiment failed: " << descriptor->id;
                if (!run_output.error_message.empty()) {
                    std::cerr << " (" << run_output.error_message << ")";
                }
                std::cerr << "\n";
                context.shutdown();
                return 1;
            }

            results.insert(results.end(), run_output.summary_results.begin(), run_output.summary_results.end());
            rows.insert(rows.end(), run_output.rows.begin(), run_output.rows.end());
        }

        const BenchmarkExportMetadata metadata{
            .gpu_name = context.selected_device_name(),
            .vulkan_api_version = format_vulkan_version(context.selected_device_api_version()),
            .driver_version = std::to_string(context.selected_device_driver_version()),
            .validation_enabled = options.enable_validation,
            .gpu_timestamps_supported = context.gpu_timestamps_supported(),
            .warmup_iterations = options.warmup_iterations,
            .timed_iterations = options.timed_iterations,
        };

        context.shutdown();

        JsonExporter::write_benchmark_results(results, rows, metadata, options.output_path);
        std::cout << "Wrote results to " << options.output_path << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Unhandled exception: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unhandled non-standard exception.\n";
        return 1;
    }
}
