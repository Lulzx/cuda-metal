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

}  // namespace cumetal::common
