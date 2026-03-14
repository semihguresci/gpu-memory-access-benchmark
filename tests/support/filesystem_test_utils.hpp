#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>

namespace TestSupport {

class ScopedTempDirectory {
  public:
    explicit ScopedTempDirectory(const std::string& prefix = "gpu_layout_tests") {
        const auto stamp =
            static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() / (prefix + "_" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~ScopedTempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    ScopedTempDirectory(const ScopedTempDirectory&) = delete;
    ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

class ScopedCurrentPath {
  public:
    explicit ScopedCurrentPath(const std::filesystem::path& path) : original_path_(std::filesystem::current_path()) {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath() {
        std::error_code ec;
        std::filesystem::current_path(original_path_, ec);
    }

    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

  private:
    std::filesystem::path original_path_;
};

} // namespace TestSupport
