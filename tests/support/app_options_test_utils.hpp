#pragma once

#include "utils/app_options.hpp"

#include <string>
#include <vector>

namespace TestSupport {

inline std::vector<char*> make_argv(std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
        argv.push_back(arg.data());
    }
    return argv;
}

inline AppOptions parse_app_options(std::vector<std::string> args,
                                    const std::vector<std::string>& available_experiment_ids) {
    std::vector<char*> argv = make_argv(args);
    return ArgumentParser::parse(static_cast<int>(argv.size()), argv.data(), available_experiment_ids);
}

} // namespace TestSupport
