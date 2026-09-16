#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace cumetal::common {

// Diagnostic wall-clock spans, not compilation-success or GPU-execution records.
// Flush begin before entering a stage so a stalled/interrupted call stays visible.
class CompileTrace {
public:
    static bool enabled() {
        const char* setting = std::getenv("CUMETAL_TRACE_COMPILE");
        return setting != nullptr && std::strcmp(setting, "1") == 0;
    }

    explicit CompileTrace(const char* stage, std::size_t input_bytes = 0)
        : stage_(stage), enabled_(enabled()) {
        if (!enabled_) return;
        span_ = next_span_.fetch_add(1, std::memory_order_relaxed);
        start_ = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "CUMETAL_COMPILE event=begin pid=%d span=%llu stage=%s input_bytes=%zu\n",
                     static_cast<int>(getpid()), static_cast<unsigned long long>(span_),
                     stage_, input_bytes);
        std::fflush(stderr);
    }

    ~CompileTrace() {
        if (!enabled_) return;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_).count();
        std::fprintf(stderr,
                     "CUMETAL_COMPILE event=end pid=%d span=%llu stage=%s elapsed_ms=%.3f\n",
                     static_cast<int>(getpid()), static_cast<unsigned long long>(span_),
                     stage_, ms);
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
};

}  // namespace cumetal::common
