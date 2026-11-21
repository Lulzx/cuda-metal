// Host-only checks for #220's compile diagnostics: the MSL function-size
// scan, CUMETAL_MSL_SIZE_LIMIT parsing and enforcement, and that timing
// stays silent unless requested.
#include "cumetal/common/compile_trace.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr const char* kSource = R"msl(// header comment with { braces }
#include <metal_stdlib>
#define CM_MACRO(x) { x }
using namespace metal;
struct Pair { float a; float b; };
static inline float helper(float x) {
    const char* s = "not a brace: {";
    return x + 1.0f; /* } */
}
namespace detail {
template <typename T> T twice(T v) { return v + v; }
}
kernel void big_kernel(device float* out [[buffer(0)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < 4) { out[i] = helper(out[i]); }
    for (uint k = 0; k < 8; ++k) { out[i] += float(k); }
    out[i] = detail::twice(out[i]);
}
)msl";

}  // namespace

int main() {
    using namespace cumetal::common;

    const auto functions = largest_msl_functions(kSource, 10);
    check(functions.size() == 3, "finds helper, twice and big_kernel (not the struct)");
    check(!functions.empty() && functions[0].name == "big_kernel", "largest is big_kernel");
    bool saw_helper = false;
    bool saw_twice = false;
    for (const auto& function : functions) {
        saw_helper = saw_helper || function.name == "helper";
        saw_twice = saw_twice || function.name == "twice";
        check(function.bytes > 0, "function sizes are positive");
    }
    check(saw_helper, "finds helper despite a brace in a string and a comment");
    check(saw_twice, "finds a template function inside a namespace");
    check(largest_msl_functions(kSource, 1).size() == 1, "respects the limit");
    check(largest_msl_functions("", 5).empty(), "empty source has no functions");

    std::size_t bytes = 0;
    check(parse_byte_size("4096", &bytes) && bytes == 4096, "plain bytes");
    check(parse_byte_size("8K", &bytes) && bytes == 8192, "K suffix");
    check(parse_byte_size("2MiB", &bytes) && bytes == 2u * 1024u * 1024u, "MiB suffix");
    check(parse_byte_size("1g", &bytes) && bytes == 1024ull * 1024ull * 1024ull, "g suffix");
    check(!parse_byte_size("", &bytes), "empty rejected");
    check(!parse_byte_size("0", &bytes), "zero rejected");
    check(!parse_byte_size("12X", &bytes), "unknown suffix rejected");
    check(!parse_byte_size("-5", &bytes), "negative rejected");
    check(format_byte_size(512) == "512 B", "bytes formatting");
    check(format_byte_size(1536) == "1.5 KiB", "KiB formatting");

    // No limit, timing off, small source: accepted silently.
    unsetenv("CUMETAL_COMPILE_TIMING");
    unsetenv("CUMETAL_MSL_SIZE_LIMIT");
    std::string error;
    check(check_metal_source_size(kSource, "probe", &error) && error.empty(),
          "small source passes without a limit");

    // Over the limit: refused, naming the limit and the largest function.
    setenv("CUMETAL_MSL_SIZE_LIMIT", "64", 1);
    error.clear();
    check(!check_metal_source_size(kSource, "probe", &error), "over-limit source refused");
    check(error.find("CUMETAL_MSL_SIZE_LIMIT=64") != std::string::npos, "error names the limit");
    check(error.find("big_kernel") != std::string::npos, "error names the largest function");
    check(error.find("probe") != std::string::npos, "error names the label");

    // Under the limit: accepted.
    setenv("CUMETAL_MSL_SIZE_LIMIT", "1M", 1);
    error.clear();
    check(check_metal_source_size(kSource, "probe", &error), "under-limit source passes");

    // A malformed limit is an error, not silently ignored.
    setenv("CUMETAL_MSL_SIZE_LIMIT", "lots", 1);
    error.clear();
    check(!check_metal_source_size(kSource, "probe", &error), "malformed limit refused");
    check(error.find("invalid CUMETAL_MSL_SIZE_LIMIT") != std::string::npos,
          "malformed limit explained");
    unsetenv("CUMETAL_MSL_SIZE_LIMIT");

    check(!compile_timing_enabled(), "timing off by default");
    setenv("CUMETAL_COMPILE_TIMING", "1", 1);
    check(compile_timing_enabled(), "CUMETAL_COMPILE_TIMING=1 enables timing");
    setenv("CUMETAL_COMPILE_TIMING", "yes", 1);
    check(!compile_timing_enabled(), "only the exact value 1 enables timing");
    unsetenv("CUMETAL_COMPILE_TIMING");

    if (failures != 0) return 1;
    std::puts("PASS: compile timing diagnostics");
    return 0;
}
