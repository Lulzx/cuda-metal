#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace cumetal::common {

// Diagnostic wall-clock spans, not compilation-success or GPU-execution records.
// An end record means scope exit, including error returns and stack unwinding.
// Flush begin before entering a stage so a stalled/interrupted call stays visible.
class CompileTrace {
public:
    static bool enabled() {
        const char* setting = std::getenv("CUMETAL_TRACE_COMPILE");
        return setting != nullptr && std::strcmp(setting, "1") == 0;
    }

    explicit CompileTrace(const char* stage, std::size_t input_bytes = 0,
                          std::string_view function = {})
        : stage_(stage), enabled_(enabled()) {
        if (!enabled_) return;
        if (!function.empty()) {
            context_ = " function=";
            // Percent-encode bytes outside a small ASCII token alphabet. This
            // keeps names reversible without injecting whitespace or records.
            constexpr char hex[] = "0123456789ABCDEF";
            for (const unsigned char c : function) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
                    context_.push_back(static_cast<char>(c));
                } else {
                    context_.push_back('%');
                    context_.push_back(hex[c >> 4]);
                    context_.push_back(hex[c & 15]);
                }
            }
        }
        span_ = next_span_.fetch_add(1, std::memory_order_relaxed);
        start_ = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "CUMETAL_COMPILE event=begin pid=%d span=%llu stage=%s input_bytes=%zu%s\n",
                     static_cast<int>(getpid()), static_cast<unsigned long long>(span_),
                     stage_, input_bytes, context_.c_str());
        std::fflush(stderr);
    }

    ~CompileTrace() {
        if (!enabled_) return;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count();
        std::fprintf(stderr,
                     "CUMETAL_COMPILE event=end pid=%d span=%llu stage=%s elapsed_ms=%.3f%s\n",
                     static_cast<int>(getpid()), static_cast<unsigned long long>(span_),
                     stage_, ms, context_.c_str());
        std::fflush(stderr);
    }

    CompileTrace(const CompileTrace&) = delete;
    CompileTrace& operator=(const CompileTrace&) = delete;

private:
    inline static std::atomic<std::uint64_t> next_span_{1};
    const char* stage_;
    bool enabled_;
    std::uint64_t span_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::string context_;
};

// Human-readable per-phase timing (CUMETAL_COMPILE_TIMING=1, or cumetalc
// --compile-timing). Independent of CUMETAL_TRACE_COMPILE, which keeps its
// machine-readable records; a CompilePhase emits both when both are enabled.
// Lines go to stderr as "cumetal timing: ...", and the process prints a
// per-phase total at exit. Implemented in compile_timing.cpp.
bool compile_timing_enabled();
// `function` names the kernel or input when known; `input_bytes` is the
// phase's input size (0 when not recorded).
void compile_phase_begin(const char* stage, std::string_view function, std::size_t input_bytes);
void compile_phase_end(const char* stage, double elapsed_ms, std::string_view function);

// One top-level compilation phase (frontend, import, MSL generation, Metal
// compile, pipeline creation). Nested sub-stages keep using CompileTrace.
class CompilePhase {
public:
    explicit CompilePhase(const char* stage, std::size_t input_bytes = 0,
                          std::string_view function = {})
        // The machine-readable record keeps its no-names contract; only the
        // opt-in human timing line names the kernel.
        : trace_(stage, input_bytes), stage_(stage),
          timing_(compile_timing_enabled()) {
        if (!timing_) return;
        function_ = std::string(function);
        compile_phase_begin(stage_, function_, input_bytes);
        start_ = std::chrono::steady_clock::now();
    }

    ~CompilePhase() {
        if (!timing_) return;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count();
        compile_phase_end(stage_, ms, function_);
    }

    CompilePhase(const CompilePhase&) = delete;
    CompilePhase& operator=(const CompilePhase&) = delete;

private:
    CompileTrace trace_;
    const char* stage_;
    bool timing_;
    std::string function_;
    std::chrono::steady_clock::time_point start_;
};

// Size of one function definition in generated Metal source.
struct MslFunctionSize {
    std::string name;
    std::size_t bytes = 0;
};

// The `limit` largest top-level function definitions in MSL text, largest
// first. A brace-depth scan that skips comments and string/char literals; it
// is a diagnostic, not a parser.
std::vector<MslFunctionSize> largest_msl_functions(std::string_view source,
                                                   std::size_t limit);

// "1.5 MiB", "12.0 KiB", "512 B".
std::string format_byte_size(std::size_t bytes);

// Parses CUMETAL_MSL_SIZE_LIMIT-style sizes: a decimal byte count with an
// optional K/M/G suffix (binary units, case-insensitive, optional trailing
// "B" or "iB"). Returns false for anything else, including zero.
bool parse_byte_size(std::string_view text, std::size_t* bytes);

// Called with generated Metal source just before it is handed to Apple's
// compiler (offline `xcrun metal` or runtime newLibraryWithSource).
// - With compile timing on, reports the source size and largest functions.
// - With CUMETAL_MSL_SIZE_LIMIT set, returns false with a diagnostic naming
//   the largest functions when the source is over the limit (or when the
//   variable is malformed), so a huge kernel fails fast instead of spending
//   minutes in Metal compilation.
// - Otherwise warns once per process for very large sources (16 MiB+).
bool check_metal_source_size(std::string_view source, std::string_view label,
                             std::string* error);

}  // namespace cumetal::common
