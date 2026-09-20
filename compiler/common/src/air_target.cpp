#include "cumetal/common/air_target.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>

namespace cumetal::common {
namespace {

constexpr const char* kFallbackTriple = "air64_v28-apple-macosx26.0.0";

std::string run_capture(const char* command) {
    std::string output;
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) {
        return output;
    }
    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output.append(buffer.data());
    }
    pclose(pipe);
    return output;
}

// Pulls "air64_v29-apple-macosx27.0.0" out of the `-###` driver dump. The
// driver prints the triple quoted on its own argument line, so we scan for the
// prefix and take the run of triple characters that follows.
std::string find_air_triple(std::string_view text) {
    constexpr std::string_view kPrefix = "air64_v";
    for (std::size_t at = text.find(kPrefix); at != std::string_view::npos;
         at = text.find(kPrefix, at + 1)) {
        std::size_t end = at;
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_' ||
                text[end] == '-' || text[end] == '.')) {
            ++end;
        }
        std::string candidate(text.substr(at, end - at));
        // Require at least the two version digits plus a vendor suffix.
        if (candidate.size() > kPrefix.size() + 2 && candidate.find('-') != std::string::npos) {
            return candidate;
        }
    }
    return {};
}

bool parse_version(const std::string& triple, int* major, int* minor) {
    constexpr std::string_view kPrefix = "air64_v";
    if (triple.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    std::string digits;
    for (std::size_t i = kPrefix.size();
         i < triple.size() && std::isdigit(static_cast<unsigned char>(triple[i])) != 0; ++i) {
        digits.push_back(triple[i]);
    }
    // Apple spells 2.9 as `v29` and 2.10 would be `v210`; the first digit is the
    // major version and the rest is the minor.
    if (digits.size() < 2) {
        return false;
    }
    *major = digits[0] - '0';
    *minor = std::stoi(digits.substr(1));
    return true;
}

AirTarget probe_air_target() {
    AirTarget target;
    target.triple = kFallbackTriple;
    parse_version(target.triple, &target.major, &target.minor);

    if (const char* override_triple = std::getenv("CUMETAL_AIR_TARGET_TRIPLE");
        override_triple != nullptr && override_triple[0] != '\0') {
        std::string forced(override_triple);
        int major = 0;
        int minor = 0;
        if (parse_version(forced, &major, &minor)) {
            target.triple = std::move(forced);
            target.major = major;
            target.minor = minor;
        }
        return target;
    }

    // `-###` only prints the commands the driver would run, so this costs a
    // process spawn and no compilation.
    const std::string dump = run_capture(
        "xcrun -sdk macosx metal -x metal -c /dev/null -o /dev/null -### 2>&1");
    const std::string triple = find_air_triple(dump);
    int major = 0;
    int minor = 0;
    if (!triple.empty() && parse_version(triple, &major, &minor)) {
        target.triple = triple;
        target.major = major;
        target.minor = minor;
    }
    return target;
}

}  // namespace

std::string AirTarget::version_string() const {
    return std::to_string(major) + "." + std::to_string(minor);
}

const AirTarget& detected_air_target() {
    static const AirTarget* target = new AirTarget(probe_air_target());
    return *target;
}

}  // namespace cumetal::common
