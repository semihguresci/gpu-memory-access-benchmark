#include "utils/app_options.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>

namespace {

bool parse_size(const std::string& value, VkDeviceSize& out_size) {
    if (value.empty()) {
        return false;
    }

    char suffix = '\0';
    std::string number_part = value;
    if (std::isdigit(static_cast<unsigned char>(value.back())) == 0) {
        suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(value.back())));
        number_part = value.substr(0, value.size() - 1);
    }

    std::size_t parsed = 0;
    unsigned long long base = 0;
    try {
        base = std::stoull(number_part, &parsed, 10);
    } catch (...) {
        return false;
    }

    if (parsed != number_part.size()) {
        return false;
    }

    unsigned long long multiplier = 1;
    if (suffix == 'K') {
        multiplier = 1024ULL;
    } else if (suffix == 'M') {
        multiplier = 1024ULL * 1024ULL;
    } else if (suffix == 'G') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    } else if (suffix != '\0') {
        return false;
    }

    out_size = static_cast<VkDeviceSize>(base * multiplier);
    return out_size > 0;
}

std::string trim_copy(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return std::string(value.substr(begin, end - begin));
}

bool is_known_experiment(const std::vector<std::string>& available_experiment_ids, std::string_view experiment_id) {
    return std::ranges::find(available_experiment_ids, experiment_id) != available_experiment_ids.end();
}

std::string format_available_experiment_ids(const std::vector<std::string>& available_experiment_ids) {
    if (available_experiment_ids.empty()) {
        return "none";
    }

    std::ostringstream stream;
    for (std::size_t index = 0; index < available_experiment_ids.size(); ++index) {
        if (index > 0) {
            stream << ", ";
        }
        stream << available_experiment_ids[index];
    }

    return stream.str();
}

std::string build_experiment_option_help(const std::vector<std::string>& available_experiment_ids) {
    return "Experiment selection: all or comma-separated IDs. Available: " +
           format_available_experiment_ids(available_experiment_ids);
}

bool parse_experiment_selection(const std::string& raw_selection,
                                const std::vector<std::string>& available_experiment_ids,
                                std::vector<std::string>& out_selected_experiment_ids, std::string& out_error) {
    out_selected_experiment_ids.clear();
    out_error.clear();

    const std::string trimmed_selection = trim_copy(raw_selection);
    if (trimmed_selection.empty()) {
        out_error = "Invalid --experiment value. Selection cannot be empty.";
        return false;
    }

    if (trimmed_selection == "all") {
        if (available_experiment_ids.empty()) {
            out_error = "No enabled experiments are registered.";
            return false;
        }

        out_selected_experiment_ids = available_experiment_ids;
        return true;
    }

    std::size_t start = 0;
    while (start <= trimmed_selection.size()) {
        const std::size_t comma = trimmed_selection.find(',', start);
        const std::size_t token_end = (comma == std::string::npos) ? trimmed_selection.size() : comma;
        const std::string token = trim_copy(std::string_view(trimmed_selection).substr(start, token_end - start));

        if (token.empty()) {
            out_error = "Invalid --experiment value. Empty token found in comma-separated list.";
            return false;
        }

        if (!is_known_experiment(available_experiment_ids, token)) {
            out_error = "Unknown experiment id: '" + token + "'.";
            return false;
        }

        if (std::ranges::find(out_selected_experiment_ids, token) == out_selected_experiment_ids.end()) {
            out_selected_experiment_ids.push_back(token);
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if (out_selected_experiment_ids.empty()) {
        out_error = "No experiment IDs were selected.";
        return false;
    }

    return true;
}

} // namespace

AppOptions ArgumentParser::parse(int argc, char** argv, const std::vector<std::string>& available_experiment_ids) {
    if (available_experiment_ids.empty()) {
        std::cerr << "No enabled experiments are registered.\n";
        std::exit(2);
    }

    AppOptions options{};
    std::string size_text = "4M";
    CLI::App app{"GPU memory layout experiments"};

    app.add_flag("--validation", options.enable_validation, "Enable Vulkan validation layers");
    app.add_option("--experiment", options.experiment, build_experiment_option_help(available_experiment_ids));
    app.add_option("--iterations", options.timed_iterations, "Timed iterations")->check(CLI::PositiveNumber);
    app.add_option("--warmup", options.warmup_iterations, "Warmup iterations")->check(CLI::NonNegativeNumber);
    app.add_flag("--verbose-progress", options.verbose_progress,
                 "Enable verbose per-stage progress logs during experiment execution");
    app.add_option(
        "--size", size_text,
        "Total scratch budget in bytes or with K/M/G suffix; experiments may split it across multiple buffers");
    app.add_option("--output", options.output_path, "Output JSON path");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    VkDeviceSize parsed_size = 0;
    if (!parse_size(size_text, parsed_size)) {
        std::cerr << "Invalid --size value. Use bytes or N[K|M|G], e.g. 4194304 or 4M.\n";
        std::exit(2);
    }
    options.scratch_size_bytes = parsed_size;

    if (options.output_path.empty()) {
        std::cerr << "Invalid --output value. Path cannot be empty.\n";
        std::exit(2);
    }

    if (!options.output_path.ends_with(".json")) {
        options.output_path += ".json";
    }

    std::string experiment_selection_error;
    if (!parse_experiment_selection(options.experiment, available_experiment_ids, options.selected_experiment_ids,
                                    experiment_selection_error)) {
        std::cerr << experiment_selection_error << "\n";
        std::cerr << "Available experiment ids: " << format_available_experiment_ids(available_experiment_ids) << "\n";
        std::exit(2);
    }

    return options;
}
