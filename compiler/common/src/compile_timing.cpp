#include "cumetal/common/compile_trace.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cumetal::common {
namespace {

constexpr const char* kPrefix = "cumetal timing:";
// Above this size Apple's compiler routinely takes minutes (vanity-miner-rs:
// 300-600 s on a 24 MB source), so say so even without timing enabled.
constexpr std::size_t kLargeSourceWarningBytes = 16u * 1024u * 1024u;
constexpr std::size_t kReportedFunctions = 5;

struct PhaseTotal {
    std::string stage;
    std::uint64_t count = 0;
    double total_ms = 0.0;
};

struct TimingState {
    std::mutex mutex;
    std::vector<PhaseTotal> totals;  // first-seen order
    bool summary_registered = false;
};

TimingState& state() {
    // Leaked deliberately: the summary runs from atexit, after static
    // destructors of other translation units may have run.
    static TimingState* instance = new TimingState();
    return *instance;
}

void print_summary() {
    TimingState& timing = state();
    std::lock_guard<std::mutex> lock(timing.mutex);
    if (timing.totals.empty()) return;
    std::fprintf(stderr, "%s summary (wall clock; phases can nest or overlap, do not add)\n",
                 kPrefix);
    for (const PhaseTotal& total : timing.totals) {
        std::fprintf(stderr, "%s   %-34s %6" PRIu64 "x %12.1f ms\n", kPrefix,
                     total.stage.c_str(), total.count, total.total_ms);
    }
    std::fflush(stderr);
}

std::string describe(std::string_view function, std::size_t input_bytes) {
    std::string detail;
    if (!function.empty()) detail += " [" + std::string(function) + "]";
    if (input_bytes != 0) detail += " (" + format_byte_size(input_bytes) + " input)";
    return detail;
}

bool is_identifier_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

bool starts_with_word(std::string_view text, std::string_view word) {
    return text.size() >= word.size() && text.substr(0, word.size()) == word &&
           (text.size() == word.size() || !is_identifier_char(text[word.size()]));
}

// Name of the function declared by `header` (text between the previous
// top-level item and its opening brace), or empty when it is not a function.
std::string function_name(std::string_view header) {
    header = trim(header);
    for (std::string_view keyword : {"struct", "class", "union", "enum", "namespace",
                                     "typedef", "using"}) {
        if (starts_with_word(header, keyword)) return {};
    }
    // Skip [[attribute(...)]] spans when looking for the parameter list.
    std::size_t paren = std::string_view::npos;
    int attribute_depth = 0;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header.compare(i, 2, "[[") == 0) {
            ++attribute_depth;
            ++i;
            continue;
        }
        if (attribute_depth > 0 && header.compare(i, 2, "]]") == 0) {
            --attribute_depth;
            ++i;
            continue;
        }
        if (attribute_depth == 0 && header[i] == '(') {
            paren = i;
            break;
        }
    }
    if (paren == std::string_view::npos) return {};
    std::size_t end = paren;
    while (end > 0 && std::isspace(static_cast<unsigned char>(header[end - 1])) != 0) --end;
    std::size_t begin = end;
    while (begin > 0 && is_identifier_char(header[begin - 1])) --begin;
    if (begin == end) {
        // operator+ and friends: keep the operator spelling.
        const std::size_t op = header.rfind("operator", end);
        if (op == std::string_view::npos) return {};
        return std::string(trim(header.substr(op, end - op)));
    }
    return std::string(header.substr(begin, end - begin));
}

}  // namespace

bool compile_timing_enabled() {
    const char* setting = std::getenv("CUMETAL_COMPILE_TIMING");
    return setting != nullptr && std::strcmp(setting, "1") == 0;
}

void compile_phase_begin(const char* stage, std::string_view function, std::size_t input_bytes) {
    std::fprintf(stderr, "%s start %s%s\n", kPrefix, stage,
                 describe(function, input_bytes).c_str());
    std::fflush(stderr);
}

void compile_phase_end(const char* stage, double elapsed_ms, std::string_view function) {
    std::fprintf(stderr, "%s done  %s%s %.1f ms\n", kPrefix, stage,
                 describe(function, 0).c_str(), elapsed_ms);
    std::fflush(stderr);
    TimingState& timing = state();
    std::lock_guard<std::mutex> lock(timing.mutex);
    auto it = std::find_if(timing.totals.begin(), timing.totals.end(),
                           [&](const PhaseTotal& total) { return total.stage == stage; });
    if (it == timing.totals.end()) {
        timing.totals.push_back({stage, 0, 0.0});
        it = timing.totals.end() - 1;
    }
    ++it->count;
    it->total_ms += elapsed_ms;
    if (!timing.summary_registered) {
        timing.summary_registered = true;
        std::atexit(print_summary);
    }
}

std::vector<MslFunctionSize> largest_msl_functions(std::string_view source,
                                                   std::size_t limit) {
    std::vector<MslFunctionSize> functions;
    // One entry per open brace: true for namespace-like scopes whose contents
    // are still "top level" for this scan.
    std::vector<bool> transparent;
    std::size_t header_start = 0;
    std::size_t function_start = std::string_view::npos;
    std::string current_name;
    std::size_t function_depth = 0;
    bool at_line_start = true;
    const auto top_level = [&] {
        return function_start == std::string_view::npos &&
               std::all_of(transparent.begin(), transparent.end(), [](bool t) { return t; });
    };
    for (std::size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (c == '\n') {
            at_line_start = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c)) != 0) continue;
        if (at_line_start && c == '#') {
            // Preprocessor line, with backslash continuations.
            while (i < source.size() && source[i] != '\n') {
                if (source[i] == '\\' && i + 1 < source.size() && source[i + 1] == '\n') ++i;
                ++i;
            }
            if (top_level()) header_start = i;
            at_line_start = true;
            continue;
        }
        at_line_start = false;
        if (c == '/' && i + 1 < source.size() && source[i + 1] == '/') {
            while (i < source.size() && source[i] != '\n') ++i;
            at_line_start = true;
            continue;
        }
        if (c == '/' && i + 1 < source.size() && source[i + 1] == '*') {
            const std::size_t close = source.find("*/", i + 2);
            i = close == std::string_view::npos ? source.size() : close + 1;
            continue;
        }
        if (c == '"' || c == '\'') {
            for (++i; i < source.size() && source[i] != c; ++i) {
                if (source[i] == '\\') ++i;
            }
            continue;
        }
        if (c == ';' && top_level()) {
            header_start = i + 1;
            continue;
        }
        if (c == '{') {
            if (top_level()) {
                const std::string_view header = source.substr(header_start, i - header_start);
                std::string name = function_name(header);
                if (!name.empty()) {
                    const std::string_view trimmed = trim(header);
                    function_start = trimmed.empty()
                                         ? i
                                         : static_cast<std::size_t>(trimmed.data() - source.data());
                    current_name = std::move(name);
                    function_depth = transparent.size();
                    transparent.push_back(false);
                    continue;
                }
                const std::string_view trimmed = trim(header);
                transparent.push_back(starts_with_word(trimmed, "namespace") ||
                                      starts_with_word(trimmed, "extern"));
                header_start = i + 1;
                continue;
            }
            transparent.push_back(false);
            continue;
        }
        if (c == '}') {
            if (transparent.empty()) continue;
            transparent.pop_back();
            if (function_start != std::string_view::npos && transparent.size() == function_depth) {
                functions.push_back({current_name, i + 1 - function_start});
                function_start = std::string_view::npos;
                header_start = i + 1;
            } else if (top_level()) {
                header_start = i + 1;
            }
        }
    }
    std::stable_sort(functions.begin(), functions.end(),
                     [](const MslFunctionSize& a, const MslFunctionSize& b) {
                         return a.bytes > b.bytes;
                     });
    if (functions.size() > limit) functions.resize(limit);
    return functions;
}

std::string format_byte_size(std::size_t bytes) {
    char buffer[64];
    if (bytes >= 1024u * 1024u) {
        std::snprintf(buffer, sizeof(buffer), "%.1f MiB",
                      static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else if (bytes >= 1024u) {
        std::snprintf(buffer, sizeof(buffer), "%.1f KiB", static_cast<double>(bytes) / 1024.0);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%zu B", bytes);
    }
    return buffer;
}

bool parse_byte_size(std::string_view text, std::size_t* bytes) {
    text = trim(text);
    std::size_t digits = 0;
    while (digits < text.size() && std::isdigit(static_cast<unsigned char>(text[digits])) != 0) {
        ++digits;
    }
    if (digits == 0 || digits > 18) return false;
    std::uint64_t value = std::stoull(std::string(text.substr(0, digits)));
    std::string suffix;
    for (char c : text.substr(digits)) {
        suffix.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    std::uint64_t scale = 1;
    if (suffix.empty() || suffix == "b") {
        scale = 1;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        scale = 1024ull;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        scale = 1024ull * 1024ull;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        scale = 1024ull * 1024ull * 1024ull;
    } else {
        return false;
    }
    if (value == 0 || value > std::numeric_limits<std::uint64_t>::max() / scale) return false;
    value *= scale;
    if (value > std::numeric_limits<std::size_t>::max()) return false;
    *bytes = static_cast<std::size_t>(value);
    return true;
}

bool check_metal_source_size(std::string_view source, std::string_view label,
                             std::string* error) {
    const bool timing = compile_timing_enabled();
    const char* limit_text = std::getenv("CUMETAL_MSL_SIZE_LIMIT");
    const bool has_limit = limit_text != nullptr && limit_text[0] != '\0';
    std::size_t limit = 0;
    if (has_limit && !parse_byte_size(limit_text, &limit)) {
        if (error != nullptr) {
            *error = std::string("invalid CUMETAL_MSL_SIZE_LIMIT '") + limit_text +
                     "': expected a positive byte count with an optional K, M or G suffix";
        }
        return false;
    }
    const bool over_limit = has_limit && source.size() > limit;
    const bool large = source.size() >= kLargeSourceWarningBytes;
    if (!timing && !over_limit && !large) return true;

    const std::string label_text = label.empty() ? std::string("kernel") : std::string(label);
    std::string largest;
    if (timing || over_limit) {
        for (const MslFunctionSize& function : largest_msl_functions(source, kReportedFunctions)) {
            if (!largest.empty()) largest += ", ";
            largest += function.name + " (" + format_byte_size(function.bytes) + ")";
        }
    }
    if (timing) {
        std::fprintf(stderr, "%s Metal source for %s: %s; largest functions: %s\n", kPrefix,
                     label_text.c_str(), format_byte_size(source.size()).c_str(),
                     largest.empty() ? "(none found)" : largest.c_str());
        std::fflush(stderr);
    }
    if (over_limit) {
        if (error != nullptr) {
            *error = "generated Metal source for " + label_text + " is " +
                     format_byte_size(source.size()) + " (" + std::to_string(source.size()) +
                     " bytes), over CUMETAL_MSL_SIZE_LIMIT=" + limit_text + " (" +
                     std::to_string(limit) + " bytes); largest functions: " +
                     (largest.empty() ? std::string("(none found)") : largest) +
                     ". Apple's Metal compile time grows with source size; see "
                     "docs/compiler-performance.md";
        }
        return false;
    }
    if (large) {
        static std::once_flag warned;
        std::call_once(warned, [&] {
            std::fprintf(stderr,
                         "cumetal: warning: generated Metal source for %s is %s; Metal "
                         "library creation can take minutes at this size. Set "
                         "CUMETAL_COMPILE_TIMING=1 for per-phase timing and the largest "
                         "functions, or CUMETAL_MSL_SIZE_LIMIT to fail fast.\n",
                         label_text.c_str(), format_byte_size(source.size()).c_str());
            std::fflush(stderr);
        });
    }
    return true;
}

}  // namespace cumetal::common
