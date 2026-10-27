#include "metal_backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

std::size_t archive_file_count(const std::filesystem::path& dir) {
    std::size_t count = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.path().extension() == ".archive") ++count;
    }
    return count;
}

bool write_text(const std::filesystem::path& path, const char* text) {
    std::ofstream out(path);
    out << text;
    return out.good();
}

}  // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "cumetal_binary_archive_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    const std::filesystem::path archive_dir = root / "archives";
    const std::filesystem::path source_one = root / "one.metal";
    const std::filesystem::path source_two = root / "two.metal";
    const char* msl_one =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void add_one(device float* x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid == 0) x[0] += 1.0f;\n}\n";
    const char* msl_two =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void add_two(device float* x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
        "    if (tid == 0) x[0] += 2.0f;\n}\n";
    bool ok = true;
    ok &= expect(write_text(source_one, msl_one), "write first MSL source");
    ok &= expect(write_text(source_two, msl_two), "write second MSL source");
    if (!ok) return 1;

    setenv("CUMETAL_BINARY_ARCHIVE_DIR", archive_dir.c_str(), 1);
    std::string error;
    ok &= expect(cumetal::metal_backend::prepare_kernel_pipeline(
                     source_one.string(), "add_one", &error) == cudaSuccess,
                 "prepare first kernel");
    if (!error.empty()) std::fprintf(stderr, "prepare error: %s\n", error.c_str());
    ok &= expect(archive_file_count(archive_dir) == 1,
                 "binary archive persisted after first pipeline creation");

    // A fresh pipeline for a different metallib reuses the cache directory
    // and adds its own archive entry.
    ok &= expect(cumetal::metal_backend::prepare_kernel_pipeline(
                     source_two.string(), "add_two", &error) == cudaSuccess,
                 "prepare second kernel");
    ok &= expect(archive_file_count(archive_dir) == 2,
                 "second metallib gets its own archive");

    // Opting out must not touch the cache directory.
    const std::filesystem::path source_three = root / "three.metal";
    ok &= expect(write_text(source_three, msl_one), "write third MSL source");
    setenv("CUMETAL_BINARY_ARCHIVE", "0", 1);
    ok &= expect(cumetal::metal_backend::prepare_kernel_pipeline(
                     source_three.string(), "add_one", &error) == cudaSuccess,
                 "prepare kernel with archive disabled");
    ok &= expect(archive_file_count(archive_dir) == 2,
                 "disabled archive writes nothing");
    unsetenv("CUMETAL_BINARY_ARCHIVE");

    std::filesystem::remove_all(root, ec);
    if (!ok) return 1;
    std::printf("PASS: binary archive cache persists and honors opt-out\n");
    return 0;
}
