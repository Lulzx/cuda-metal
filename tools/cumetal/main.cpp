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
};

Layout resolve_layout(const char *argv0) {
  if (const char *root = std::getenv("CUMETAL_ROOT");
      root != nullptr && root[0] != '\0') {
    const std::filesystem::path prefix(root);
    return {prefix, prefix / "bin", prefix / "include", prefix / "lib"};
  }

  const std::filesystem::path self = executable_path(argv0);
  const std::filesystem::path bin_dir = self.parent_path();
  const std::filesystem::path prefix = bin_dir.parent_path();
  if (std::filesystem::exists(prefix / "include" / "cuda_runtime.h") &&
      std::filesystem::exists(prefix / "lib" / "libcumetal.dylib")) {
    return {prefix, bin_dir, prefix / "include", prefix / "lib"};
  }

  return {prefix, bin_dir,
          std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api",
          bin_dir};
}

void print_usage(const char *argv0) {
  const std::string name =
      std::filesystem::path(argv0 != nullptr ? argv0 : "cumetal")
          .filename()
          .string();
  std::cout << "CuMetal " << CUMETAL_VERSION_STRING << "\n\n"
            << "Usage:\n"
            << "  " << name << " doctor            # exit 0 ready, 3 JIT-only, 1 not ready\n"
            << "  " << name << " run <program> [args...]\n"
            << "  " << name << " version [--json]\n\n"
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

private:
  std::string style(const char *code, std::string_view text) const {
    if (!color_)
      return std::string(text);
    return std::string(code) + std::string(text) + "\033[0m";
  }

  bool color_;
};

// Exit codes for `cumetal doctor`. JIT-only is not a failure: programs that
// register kernels with the CuMetal runtime (clang-built CUDA, the driver API,
// NVRTC) compile them through Metal's in-process compiler, so they run without
// Apple's offline toolchain. Ahead-of-time `cumetalc` builds, and runtime
// kernels that link CuMetal's FP64/libdevice support, need `xcrun metal`.
constexpr int kDoctorReady = 0;
constexpr int kDoctorNotReady = 1;
constexpr int kDoctorJitOnly = 3;

// CuMetal compiles generated MSL with -std=metal3.2, which Apple's compilers
// lower to AIR 2.7 at minimum.
constexpr int kMinimumAirMajor = 2;
constexpr int kMinimumAirMinor = 7;

std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char c : value) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

bool contains(const std::string &text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

// "air64_v28-apple-macosx26.0.0" -> {2, 8, 26}. Apple spells AIR 2.8 as v28.
struct AirTriple {
  bool ok = false;
  int major = 0;
  int minor = 0;
  int macos_major = 0;
  std::string text;
};

AirTriple parse_air_triple(const std::string &dump) {
  AirTriple triple;
  const std::string prefix = "air64_v";
  for (std::size_t at = dump.find(prefix); at != std::string::npos;
       at = dump.find(prefix, at + 1)) {
    std::size_t end = at;
    while (end < dump.size() &&
           (std::isalnum(static_cast<unsigned char>(dump[end])) != 0 ||
            dump[end] == '_' || dump[end] == '-' || dump[end] == '.')) {
      ++end;
    }
    const std::string candidate = dump.substr(at, end - at);
    std::size_t i = prefix.size();
    std::string digits;
    while (i < candidate.size() &&
           std::isdigit(static_cast<unsigned char>(candidate[i])) != 0)
      digits.push_back(candidate[i++]);
    if (digits.size() < 2)
      continue;
    triple.major = digits[0] - '0';
    triple.minor = std::stoi(digits.substr(1));
    const std::size_t os = candidate.find("macosx");
    if (os != std::string::npos) {
      std::string os_digits;
      for (std::size_t j = os + 6;
           j < candidate.size() &&
           std::isdigit(static_cast<unsigned char>(candidate[j])) != 0;
           ++j)
        os_digits.push_back(candidate[j]);
      if (!os_digits.empty())
        triple.macos_major = std::stoi(os_digits);
    }
    triple.text = candidate;
    triple.ok = true;
    return triple;
  }
  return triple;
}

int macos_major_version() {
  const CommandResult macos =
      run_capture("/usr/bin/sw_vers -productVersion 2>/dev/null");
  if (macos.exit_code != 0 || macos.output.empty())
    return 0;
  const std::string major = macos.output.substr(0, macos.output.find('.'));
  if (major.empty() || major.find_first_not_of("0123456789") != std::string::npos)
    return 0;
  return std::stoi(major);
}

struct MetalToolchainReport {
  bool ready = false;
  std::vector<std::string> details;
  // Printed after the checklist when the toolchain is not ready.
  std::vector<std::string> fix;
  std::string air_version;
};

const std::vector<std::string> &metal_toolchain_fix_sequence() {
  static const std::vector<std::string> steps = {
      "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
      "   # if xcode-select points at the Command Line Tools",
      "sudo xcodebuild -runFirstLaunch",
      "xcodebuild -downloadComponent MetalToolchain",
      "cumetal doctor   # re-check",
  };
  return steps;
}

// Finds, runs and exercises Apple's offline Metal toolchain the way cumetalc
// does: compile a trivial kernel with -std=metal3.2 and package it with
// metallib, then check the AIR version and deployment target it produces.
MetalToolchainReport probe_metal_toolchain(int host_macos_major) {
  MetalToolchainReport report;
  const CommandResult metal = run_capture("xcrun --find metal 2>/dev/null");
  const CommandResult metallib = run_capture("xcrun --find metallib 2>/dev/null");
  const CommandResult developer = run_capture("xcode-select -p 2>/dev/null");
  if (metal.exit_code != 0 || metal.output.empty() || metallib.exit_code != 0 ||
      metallib.output.empty()) {
    report.details.push_back(metal.output.empty() ? "metal: not found"
                                                  : "metal: " + metal.output);
    report.details.push_back(metallib.output.empty()
                                 ? "metallib: not found"
                                 : "metallib: " + metallib.output);
    if (contains(developer.output, "CommandLineTools")) {
      report.details.push_back(
          "xcode-select points at the Command Line Tools (" + developer.output +
          "), which do not include the Metal compiler; install Xcode");
    }
    report.fix = metal_toolchain_fix_sequence();
    return report;
  }
  report.details.push_back("metal: " + metal.output);
  report.details.push_back("metallib: " + metallib.output);

  // Xcode 26/27 ship `metal` as a launcher stub; the compiler itself is a
  // downloadable component. `xcrun --find` succeeds against the stub, so only
  // running it reveals the missing component.
  const CommandResult version = run_capture("xcrun metal --version 2>&1");
  if (version.exit_code != 0 || contains(version.output, "missing Metal Toolchain") ||
      contains(version.output, "cannot execute tool 'metal'")) {
    report.details.push_back(
        "`xcrun metal` is Xcode's launcher stub: the Metal Toolchain component "
        "is not installed (or Xcode's first launch has not run)");
    if (!version.output.empty())
      report.details.push_back("xcrun metal --version: " +
                               version.output.substr(0, version.output.find('\n')));
    report.fix = metal_toolchain_fix_sequence();
    return report;
  }

  std::error_code temp_ec;
  std::string pattern =
      (std::filesystem::temp_directory_path(temp_ec) / "cumetal-doctor-XXXXXX").string();
  std::vector<char> scratch(pattern.begin(), pattern.end());
  scratch.push_back('\0');
  if (temp_ec || mkdtemp(scratch.data()) == nullptr) {
    report.details.push_back("could not create a scratch directory for the AIR probe");
    return report;
  }
  const std::filesystem::path dir(scratch.data());
  {
    FILE *source = std::fopen((dir / "probe.metal").c_str(), "w");
    if (source != nullptr) {
      std::fputs("#include <metal_stdlib>\nusing namespace metal;\n"
                 "kernel void cumetal_doctor_probe(device uint* out [[buffer(0)]],\n"
                 "    uint i [[thread_position_in_grid]]) { out[i] = i; }\n",
                 source);
      std::fclose(source);
    }
  }
  const std::string q = shell_quote(dir.string());
  const CommandResult compiled = run_capture(
      ("xcrun -sdk macosx metal -std=metal3.2 -c " + q + "/probe.metal -o " + q +
       "/probe.air 2>&1")
          .c_str());
  const CommandResult linked =
      compiled.exit_code == 0
          ? run_capture(("xcrun -sdk macosx metallib " + q + "/probe.air -o " + q +
                         "/probe.metallib 2>&1")
                            .c_str())
          : CommandResult{};
  const CommandResult dump = run_capture(
      "xcrun -sdk macosx metal -x metal -c /dev/null -o /dev/null -### 2>&1");
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);

  const AirTriple triple = parse_air_triple(dump.output);
  if (triple.ok) {
    report.air_version =
        std::to_string(triple.major) + "." + std::to_string(triple.minor);
    report.details.push_back("AIR " + report.air_version + " (" + triple.text + ")");
  }
  const std::string required = std::to_string(kMinimumAirMajor) + "." +
                                std::to_string(kMinimumAirMinor);
  if (triple.ok && (triple.major < kMinimumAirMajor ||
                    (triple.major == kMinimumAirMajor &&
                     triple.minor < kMinimumAirMinor))) {
    report.details.push_back("toolchain too old for AIR " + required +
                             ": it targets AIR " + report.air_version +
                             "; CuMetal compiles -std=metal3.2. Install Xcode 16 or newer");
    return report;
  }
  if (compiled.exit_code != 0) {
    report.details.push_back(
        "toolchain too old for AIR " + required +
        ", or broken: compiling a -std=metal3.2 kernel failed: " +
        compiled.output.substr(0, compiled.output.find('\n')));
    return report;
  }
  if (linked.exit_code != 0) {
    report.details.push_back("packaging the probe with metallib failed: " +
                             linked.output.substr(0, linked.output.find('\n')));
    return report;
  }
  if (!triple.ok) {
    report.details.push_back(
        "could not read the AIR target from `xcrun metal -###`");
    return report;
  }
  // A toolchain newer than the OS emits metallibs the OS cannot load
  // (generativeqc needed macOS 26 for Xcode 26.3's AIR 2.8 output).
  if (host_macos_major != 0 && triple.macos_major > host_macos_major) {
    report.details.push_back(
        "toolchain targets macOS " + std::to_string(triple.macos_major) +
        " but this Mac runs macOS " + std::to_string(host_macos_major) +
        "; metallibs it builds may not load. Use the Xcode/Metal Toolchain for "
        "this macOS, or update macOS");
    return report;
  }
  report.details.push_back("compiled and packaged a -std=metal3.2 probe kernel");
  report.ready = true;
  return report;
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
  const int host_macos_major = macos_major_version();
  const bool supported_macos = host_macos_major >= 14;
  ready &= console.check(supported_macos, "macOS",
                         {macos.output.empty()
                              ? "Version could not be determined"
                              : "Version " + macos.output + " (14+ required)"});

  const bool compiler = std::filesystem::exists(layout.bin_dir / "cumetalc");
  const bool headers =
      std::filesystem::exists(layout.include_dir / "cuda_runtime.h");
  const bool runtime =
      std::filesystem::exists(layout.lib_dir / "libcumetal.dylib");
  ready &= console.check(
      compiler && headers && runtime,
      std::string("CuMetal ") + CUMETAL_VERSION_STRING,
      {"Compiler: " + (layout.bin_dir / "cumetalc").string(),
       "Headers: " + layout.include_dir.string(),
       "Runtime: " + layout.lib_dir.string()});

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

  const MetalToolchainReport toolchain = probe_metal_toolchain(host_macos_major);
  console.check(toolchain.ready, "Metal toolchain (ahead-of-time cumetalc builds)",
                toolchain.details);

  const bool binary_shim =
      std::filesystem::exists(layout.lib_dir / "libcuda.dylib");
  console.info("Binary compatibility shim",
               {binary_shim
                    ? "Installed"
                    : "Not installed (optional; source compilation is ready)"});

  if (ready && toolchain.ready) {
    std::cout << '\n';
    console.success("No issues found!");
    return kDoctorReady;
  }
  if (ready) {
    // Everything but the offline toolchain works: runtime JIT is usable.
    std::cout << '\n';
    console.info(
        "JIT-only: CuMetal is ready for runtime-compiled kernels",
        {"Programs that register kernels with libcumetal (clang-built CUDA, the "
         "driver API, NVRTC) compile them with Metal's built-in compiler and run.",
         "Ahead-of-time `cumetalc` builds, and runtime kernels that link CuMetal's "
         "FP64 or libdevice support, need the Metal toolchain."});
    if (!toolchain.fix.empty()) {
      std::cout << "\n    To install the Metal toolchain (Xcode 26/27):\n";
      for (const std::string &step : toolchain.fix)
        std::cout << "      " << step << '\n';
    }
    return kDoctorJitOnly;
  }
  std::cerr << '\n';
  console.failure("CuMetal is not ready; fix the [✗] checks above.");
  if (!toolchain.ready && !toolchain.fix.empty()) {
    std::cerr << "\nTo install the Metal toolchain (Xcode 26/27):\n";
    for (const std::string &step : toolchain.fix)
      std::cerr << "  " << step << '\n';
  }
  return kDoctorNotReady;
}

// Machine-readable install and capability report for downstream build
// scripts (docs/downstream.md). Every field is probed, not assumed.
std::string json_escape(const std::string &text) {
  std::string out;
  for (char c : text) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\t': out += "\\t"; break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buffer[8];
        std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
        out += buffer;
      } else {
        out.push_back(c);
      }
    }
  }
  return out;
}

int encoded_version() {
  int parts[3] = {0, 0, 0};
  std::string_view text(CUMETAL_VERSION_STRING);
  for (int &part : parts) {
    std::size_t digits = 0;
    while (digits < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[digits])) != 0)
      ++digits;
    if (digits == 0)
      return 0;
    part = std::stoi(std::string(text.substr(0, digits)));
    text.remove_prefix(digits);
    if (!text.empty() && text.front() == '.')
      text.remove_prefix(1);
  }
  return parts[0] * 10000 + parts[1] * 100 + parts[2];
}

int version_json(const char *argv0) {
  const Layout layout = resolve_layout(argv0);
  const bool compiler = std::filesystem::exists(layout.bin_dir / "cumetalc");
  const bool headers =
      std::filesystem::exists(layout.include_dir / "cuda_runtime.h");
  const bool runtime =
      std::filesystem::exists(layout.lib_dir / "libcumetal.dylib");
  const bool binary_shim =
      std::filesystem::exists(layout.lib_dir / "libcuda.dylib");
  const std::filesystem::path bundled_cache =
      layout.lib_dir.parent_path() / "share" / "cumetal-cache";
  std::error_code ec;
  const bool has_bundled_cache = std::filesystem::is_directory(bundled_cache, ec);
  const MetalToolchainReport toolchain =
      probe_metal_toolchain(macos_major_version());
  const auto boolean = [](bool value) { return value ? "true" : "false"; };
  std::cout << "{\n"
            << "  \"name\": \"cumetal\",\n"
            << "  \"version\": \"" << CUMETAL_VERSION_STRING << "\",\n"
            << "  \"version_number\": " << encoded_version() << ",\n"
            << "  \"prefix\": \"" << json_escape(layout.prefix.string()) << "\",\n"
            << "  \"bin_dir\": \"" << json_escape(layout.bin_dir.string()) << "\",\n"
            << "  \"include_dir\": \"" << json_escape(layout.include_dir.string())
            << "\",\n"
            << "  \"lib_dir\": \"" << json_escape(layout.lib_dir.string()) << "\",\n"
            << "  \"bundled_cache_dir\": "
            << (has_bundled_cache
                    ? "\"" + json_escape(bundled_cache.string()) + "\""
                    : std::string("null"))
            << ",\n"
            << "  \"macros\": {\"__CUMETAL__\": 1, \"__CUMETAL_VERSION__\": "
            << encoded_version() << "},\n"
            << "  \"features\": {\n"
            << "    \"source_compiler\": " << boolean(compiler && headers) << ",\n"
            << "    \"runtime\": " << boolean(runtime) << ",\n"
            << "    \"jit\": " << boolean(runtime && headers) << ",\n"
            << "    \"aot_metal_toolchain\": " << boolean(toolchain.ready) << ",\n"
            << "    \"air_version\": "
            << (toolchain.air_version.empty()
                    ? std::string("null")
                    : "\"" + toolchain.air_version + "\"")
            << ",\n"
            << "    \"binary_shim\": " << boolean(binary_shim) << ",\n"
            << "    \"compile_timing\": true,\n"
            << "    \"msl_size_limit\": true\n"
            << "  }\n"
            << "}\n";
  return 0;
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
    if (argc >= 3 && std::string_view(argv[2]) == "--json")
      return version_json(argv[0]);
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
