#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

struct AppOptions {
#if defined(DEFAULT_ENABLE_VULKAN_VALIDATION)
    bool enable_validation = true;
#else
    bool enable_validation = false;
#endif
    std::string experiment = "all";
    std::vector<std::string> selected_experiment_ids;
    int timed_iterations = 20;
    int warmup_iterations = 5;
    VkDeviceSize scratch_size_bytes = static_cast<VkDeviceSize>(4ULL * 1024ULL * 1024ULL);
    std::string output_path = "results/tables/benchmark_results.json";
};

class ArgumentParser {
  public:
    static AppOptions parse(int argc, char** argv, const std::vector<std::string>& available_experiment_ids);
};
