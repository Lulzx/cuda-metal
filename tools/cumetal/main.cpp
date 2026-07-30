#include <mach-o/dyld.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifndef CUMETAL_SOURCE_DIR
#define CUMETAL_SOURCE_DIR ""
#endif

#ifndef CUMETAL_VERSION_STRING
#define CUMETAL_VERSION_STRING "unknown"
#endif

namespace {

struct CommandResult {
  int exit_code = 1;
  std::string output;
};

CommandResult run_capture(const char *command) {
  CommandResult result;
  std::array<char, 512> buffer{};
  FILE *pipe = popen(command, "r");
  if (pipe == nullptr)
    return result;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
         nullptr) {
    result.output.append(buffer.data());
  }
  const int status = pclose(pipe);
  if (WIFEXITED(status))
    result.exit_code = WEXITSTATUS(status);
  while (!result.output.empty() &&
         std::isspace(static_cast<unsigned char>(result.output.back())) != 0) {
    result.output.pop_back();
  }
  return result;
}

std::filesystem::path executable_path(const char *argv0) {
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> buffer(size + 1, '\0');
  std::error_code ec;
  if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
    const auto resolved = std::filesystem::weakly_canonical(buffer.data(), ec);
    if (!ec)
      return resolved;
  }
  if (argv0 != nullptr && argv0[0] != '\0') {
    const auto resolved = std::filesystem::weakly_canonical(argv0, ec);
    if (!ec)
      return resolved;
  }
  return {};
}

struct Layout {
  std::filesystem::path prefix;
  std::filesystem::path bin_dir;
  std::filesystem::path include_dir;
  std::filesystem::path lib_dir;
  std::filesystem::path vector_add_example;
};

Layout resolve_layout(const char *argv0) {
  if (const char *root = std::getenv("CUMETAL_ROOT");
      root != nullptr && root[0] != '\0') {
    const std::filesystem::path prefix(root);
    return {prefix, prefix / "bin", prefix / "include", prefix / "lib",
            prefix / "share" / "cumetal" / "examples" / "vectorAdd.cu"};
  }

  const std::filesystem::path self = executable_path(argv0);
  const std::filesystem::path bin_dir = self.parent_path();
  const std::filesystem::path prefix = bin_dir.parent_path();
  if (std::filesystem::exists(prefix / "include" / "cuda_runtime.h") &&
      std::filesystem::exists(prefix / "lib" / "libcumetal.dylib")) {
    return {prefix, bin_dir, prefix / "include", prefix / "lib",
            prefix / "share" / "cumetal" / "examples" / "vectorAdd.cu"};
  }

  return {prefix, bin_dir,
          std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api",
          bin_dir,
          std::filesystem::path(CUMETAL_SOURCE_DIR) / "samples" / "vectorAdd" /
              "vectorAdd.cu"};
}

void print_usage(const char *argv0) {
  const std::string name =
      std::filesystem::path(argv0 != nullptr ? argv0 : "cumetal")
          .filename()
          .string();
  std::cout << "CuMetal " << CUMETAL_VERSION_STRING << "\n\n"
            << "Usage:\n"
            << "  " << name << " doctor\n"
            << "  " << name << " run <program> [args...]\n"
            << "  " << name << " version\n\n"
            << "Compile CUDA source with:\n"
            << "  cumetalc program.cu -o program\n";
}

bool color_enabled() {
  if (std::getenv("NO_COLOR") != nullptr)
    return false;
  if (const char *forced = std::getenv("CLICOLOR_FORCE");
      forced != nullptr && forced[0] != '\0' &&
      std::string_view(forced) != "0") {
    return true;
  }
  if (const char *term = std::getenv("TERM");
      term != nullptr && std::string_view(term) == "dumb") {
    return false;
  }
  return isatty(STDOUT_FILENO) != 0;
}

class DoctorConsole {
public:
  DoctorConsole() : color_(color_enabled()) {}

  void heading() const {
    std::cout << style("\033[1;36m", "CuMetal doctor") << "\n\n";
  }

  bool check(bool ok, std::string_view label,
             const std::vector<std::string> &details = {}) const {
    const char *color = ok ? "\033[32m" : "\033[31m";
    const std::string marker = ok ? "[✓]" : "[✗]";
    std::cout << style(color, marker) << " " << style("\033[1m", label) << '\n';
    for (const std::string &detail : details)
      std::cout << "    " << style("\033[2m", "•") << " " << detail << '\n';
    return ok;
  }

  void info(std::string_view label,
            const std::vector<std::string> &details = {}) const {
    std::cout << style("\033[36m", "[•]") << " " << style("\033[1m", label)
              << '\n';
    for (const std::string &detail : details)
      std::cout << "    " << style("\033[2m", "•") << " " << detail << '\n';
  }

  void success(std::string_view message) const {
    std::cout << style("\033[1;32m", message) << '\n';
  }

  void failure(std::string_view message) const {
    std::cerr << (color_ ? "\033[1;31m" : "") << message
              << (color_ ? "\033[0m" : "") << '\n';
  }

  void command(std::string_view command) const {
    std::cout << "  " << style("\033[36m", command) << '\n';
  }

private:
  std::string style(const char *code, std::string_view text) const {
    if (!color_)
      return std::string(text);
    return std::string(code) + std::string(text) + "\033[0m";
  }

  bool color_;
};

std::string shell_quote(std::string_view value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'')
      quoted += "'\\''";
    else
      quoted += ch;
  }
  quoted += "'";
  return quoted;
}

int doctor(const char *argv0) {
  const Layout layout = resolve_layout(argv0);
  const DoctorConsole console;
  bool ready = true;

  console.heading();
  std::cout
      << "Doctor summary (all required components are checked below):\n\n";

  const CommandResult arch = run_capture("/usr/bin/uname -m 2>/dev/null");
  ready &= console.check(
      arch.exit_code == 0 && arch.output == "arm64", "Apple Silicon",
      {arch.output.empty() ? "Architecture could not be determined"
                           : "Architecture: " + arch.output});

  const CommandResult macos =
      run_capture("/usr/bin/sw_vers -productVersion 2>/dev/null");
  bool supported_macos = false;
  if (macos.exit_code == 0 && !macos.output.empty()) {
    const std::size_t dot = macos.output.find('.');
    const std::string major = macos.output.substr(0, dot);
    supported_macos =
        !major.empty() &&
        major.find_first_not_of("0123456789") == std::string::npos &&
        std::stoi(major) >= 14;
  }
  ready &= console.check(supported_macos, "macOS",
                         {macos.output.empty()
                              ? "Version could not be determined"
                              : "Version " + macos.output + " (14+ required)"});

  const bool compiler = std::filesystem::exists(layout.bin_dir / "cumetalc");
  const bool headers =
      std::filesystem::exists(layout.include_dir / "cuda_runtime.h");
  const bool runtime =
      std::filesystem::exists(layout.lib_dir / "libcumetal.dylib");
  const bool example = std::filesystem::exists(layout.vector_add_example);
  ready &=
      console.check(compiler && headers && runtime && example,
                    std::string("CuMetal ") + CUMETAL_VERSION_STRING,
                    {"Compiler: " + (layout.bin_dir / "cumetalc").string(),
                     "Headers: " + layout.include_dir.string(),
                     "Runtime: " + layout.lib_dir.string(),
                     "Bundled example: " + layout.vector_add_example.string()});

  std::filesystem::path clang;
  if (const char *configured = std::getenv("CUMETAL_CUDA_CLANG");
      configured != nullptr && configured[0] != '\0') {
    clang = configured;
  } else {
    static constexpr const char *candidates[] = {
        "/opt/homebrew/opt/llvm/bin/clang++",
        "/usr/local/opt/llvm/bin/clang++",
    };
    for (const char *candidate : candidates) {
      if (std::filesystem::exists(candidate)) {
        clang = candidate;
        break;
      }
    }
    if (clang.empty()) {
      const CommandResult found = run_capture("command -v clang++ 2>/dev/null");
      if (found.exit_code == 0)
        clang = found.output;
    }
  }
  ready &= console.check(
      !clang.empty() && std::filesystem::exists(clang), "CUDA-capable Clang",
      {clang.empty() ? "Not found; install with `brew install llvm`"
                     : clang.string()});

  const CommandResult metal = run_capture("xcrun --find metal 2>/dev/null");
  const CommandResult metallib =
      run_capture("xcrun --find metallib 2>/dev/null");
  const bool metal_ready = metal.exit_code == 0 && !metal.output.empty() &&
                           metallib.exit_code == 0 && !metallib.output.empty();
  ready &= console.check(
      metal_ready, "Metal toolchain",
      {metal.output.empty() ? "metal: not found" : "metal: " + metal.output,
       metallib.output.empty() ? "metallib: not found"
                               : "metallib: " + metallib.output});

  const bool binary_shim =
      std::filesystem::exists(layout.lib_dir / "libcuda.dylib");
  console.info("Binary compatibility shim",
               {binary_shim
                    ? "Installed"
                    : "Not installed (optional; source compilation is ready)"});

  if (ready) {
    std::cout << '\n';
    console.success("No issues found!");
    std::cout << "\nTry the bundled vectorAdd example:\n";
    console.command("cumetalc " +
                    shell_quote(layout.vector_add_example.string()) +
                    " -o /tmp/vectorAdd");
    console.command("/tmp/vectorAdd");
    return 0;
  }
  std::cerr << '\n';
  console.failure("CuMetal is not ready; fix the [✗] checks above.");
  return 1;
}

std::string prepend_path(const std::filesystem::path &directory,
                         const char *existing) {
  if (existing == nullptr || existing[0] == '\0')
    return directory.string();
  return directory.string() + ":" + existing;
}

int run_program(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "cumetal run expects a program path\n";
    return 2;
  }

  const Layout layout = resolve_layout(argv[0]);
  if (!std::filesystem::exists(layout.lib_dir / "libcumetal.dylib")) {
    std::cerr << "cumetal run failed: libcumetal.dylib was not found under "
              << layout.lib_dir
              << "\n"
                 "Set CUMETAL_ROOT to the CuMetal installation prefix.\n";
    return 1;
  }

  const std::string dyld_library =
      prepend_path(layout.lib_dir, std::getenv("DYLD_LIBRARY_PATH"));
  const std::string dyld_fallback =
      prepend_path(layout.lib_dir, std::getenv("DYLD_FALLBACK_LIBRARY_PATH"));
  setenv("DYLD_LIBRARY_PATH", dyld_library.c_str(), 1);
  setenv("DYLD_FALLBACK_LIBRARY_PATH", dyld_fallback.c_str(), 1);

  execvp(argv[2], &argv[2]);
  std::perror("cumetal run failed");
  return 127;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 0;
  }

  const std::string_view command(argv[1]);
  if (command == "doctor")
    return doctor(argv[0]);
  if (command == "run")
    return run_program(argc, argv);
  if (command == "version" || command == "--version" || command == "-v") {
    std::cout << "cumetal " << CUMETAL_VERSION_STRING << '\n';
    return 0;
  }
  if (command == "help" || command == "--help" || command == "-h") {
    print_usage(argv[0]);
    return 0;
  }

  std::cerr << "unknown command: " << command << "\n\n";
  print_usage(argv[0]);
  return 2;
}
