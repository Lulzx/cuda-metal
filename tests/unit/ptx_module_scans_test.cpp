#include "ptx_module.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Keep the original whole-source expressions as the regression oracle. The
// optimized scanners must preserve their captures and raw-source behavior.
const std::array<std::regex, 5> legacy_patterns = {
    std::regex(R"((?:\.extern\s+)?\.shared\s+\.align\s+([0-9]+)\s+\.(?:b|u|s|f)(8|16|32|64)\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*(?:\[\s*([0-9]*)\s*\])?\s*;)"),
    std::regex(R"(\.local\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"),
    std::regex(R"(//\s*implicit-def:\s*(%[A-Za-z0-9_.$]+))"),
    std::regex(R"((?:\.visible\s+|\.extern\s+)?\.const\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"),
    std::regex(R"((?:\.visible\s+|\.extern\s+)?\.global\s+\.align\s+([0-9]+)\s+\.b8\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\[\s*([0-9]+)\s*\]\s*;)"),
};

struct Scans {
    // Rows encode every returned field, retaining declaration order and duplicates.
    std::vector<std::string> shared, local, constants, globals;
    std::set<std::string> implicit;
    bool operator==(const Scans&) const = default;
};

std::string row(const std::string& name, std::uint64_t bytes,
                std::uint32_t alignment, std::uint64_t offset = 0) {
    return name + ":" + std::to_string(bytes) + ":" +
           std::to_string(alignment) + ":" + std::to_string(offset);
}

Scans legacy_scans(std::string_view ptx) {
    const std::string source(ptx);
    Scans out;
    for (std::size_t kind = 0; kind < legacy_patterns.size(); ++kind) {
        std::uint64_t cursor = 0;
        for (std::sregex_iterator it(source.begin(), source.end(), legacy_patterns[kind]), end;
             it != end; ++it) {
            const auto& match = *it;
            if (kind == 2) {
                out.implicit.insert(match[1].str());
                continue;
            }
            const auto alignment = static_cast<std::uint32_t>(std::stoul(match[1].str()));
            if (kind == 0) {
                const bool dynamic = match[4].matched && match[4].str().empty();
                const std::uint64_t count = !match[4].matched ? 1 :
                    (dynamic ? 0 : std::stoull(match[4].str()));
                out.shared.push_back(row(match[3].str(),
                    std::stoull(match[2].str()) / 8 * count, alignment) +
                    (dynamic ? ":dynamic" : ":static"));
            } else {
                const std::uint64_t bytes = std::stoull(match[3].str());
                if (kind == 3) cursor = (cursor + alignment - 1) / alignment * alignment;
                auto& rows = kind == 1 ? out.local : kind == 3 ? out.constants : out.globals;
                rows.push_back(row(match[2].str(), bytes, alignment, kind == 3 ? cursor : 0));
                if (kind == 3) cursor += bytes;
            }
        }
    }
    return out;
}

Scans optimized_scans(std::string_view ptx) {
    using namespace cumetal::ir::detail;
    Scans out;
    for (const auto& item : scan_threadgroup_globals(ptx))
        out.shared.push_back(row(item.name, item.byte_size, item.alignment) +
                             (item.is_dynamic ? ":dynamic" : ":static"));
    for (const auto& item : scan_local_depots(ptx))
        out.local.push_back(row(item.name, item.byte_size, item.alignment));
    for (const auto& item : scan_implicit_definitions(ptx)) out.implicit.insert(item);
    for (const auto& item : scan_module_constant_symbols(ptx))
        out.constants.push_back(row(item.name, item.byte_size, item.alignment, item.offset));
    for (const auto& item : scan_module_global_symbols(ptx))
        out.globals.push_back(row(item.name, item.byte_size, item.alignment, item.offset));
    return out;
}

bool compare(std::string_view label, std::string_view source) {
    if (optimized_scans(source) == legacy_scans(source)) return true;
    std::cerr << "FAIL: module scans changed for " << label << '\n';
    return false;
}

}  // namespace

int main() {
    bool ok = compare("empty", {});
    const std::string declarations = R"PTX(
.shared .align 1 .b8 byte$tile[3];
.shared .align 2 .u16 unsigned_tile[5];
.shared .align 4 .s32 signed_tile[7];
.extern .shared .align 8 .f64 scalar;
.extern .shared .align 16 .b8 dynamic[];
.shared .align 4 .u32 zero[0];
.local .align 32 .b8 depot$0[128]; .local .align 8 .b8 depot$0[16];
.const .align 1 .b8 c0[3]; .visible .const .align 8 .b8 c1[5];
.extern .const .align 16 .b8 c2[17]; .const .align 4 .b8 c2[1];
.visible .global .align 32 .b8 global$0[35];
.extern .global .align 8 .b8 global$1[9]; .global .align 1 .b8 global$0[2];
// implicit-def: %r0
// implicit-def: %r0
/// implicit-def: %overlap$1
// implicit-def: %.dot
)PTX";
    ok &= compare("all captures, optional qualifiers, duplicates and overlapping markers", declarations);
    const auto basic = optimized_scans(declarations);
    ok &= basic.shared == std::vector<std::string>{
        "byte$tile:3:1:0:static", "unsigned_tile:10:2:0:static",
        "signed_tile:28:4:0:static", "scalar:8:8:0:static",
        "dynamic:0:16:0:dynamic", "zero:0:4:0:static"};
    ok &= basic.local == std::vector<std::string>{"depot$0:128:32:0", "depot$0:16:8:0"};
    ok &= basic.constants == std::vector<std::string>{
        "c0:3:1:0", "c1:5:8:8", "c2:17:16:16", "c2:1:4:36"};
    ok &= basic.globals == std::vector<std::string>{
        "global$0:35:32:0", "global$1:9:8:0", "global$0:2:1:0"};
    ok &= basic.implicit == std::set<std::string>{"%r0", "%overlap$1", "%.dot"};

    const std::string false_starts = R"PTX(
ld.shared.u32 %r0, [%rd0]; st.local.b8 [%rd1], 0;
ld.const.b32 %r1, [c0]; st.global.u64 [%rd2], %rd3;
.sharedX .localX .constX .globalX // implicit-defX: %ignored
.shared .align 8 .b128 bad[1]; .local .align 8 .u8 bad[1];
.const .align 8 .b8 missing_extent; .global .align 8 .b8 missing_semicolon[1]
.local .align 8 .b8 broken[ .local .align 4 .b8 recovered[2];
.shared .align x .u32 bad[1]; .shared .align 2 .u16 recovered[3];
.const .align 8 .b8 initialized[1] = {0}; .global .align 8 .b8 initialized[1] = {0};
.const .align 1 .b8 ok[2]; .global .align 1 .b8 ok[2];
// ordinary comment // implicit-def: %recovered
// implicit-def: bad // implicit-def: %good
)PTX";
    ok &= compare("instruction suffixes and malformed candidates before valid declarations", false_starts);
    ok &= compare("valid declarations after false starts", false_starts + declarations);
    std::string multiline;
    for (const char c : declarations) multiline += c == ' ' ? "\r\n\t" : std::string(1, c);
    ok &= compare("CRLF and multiline whitespace including implicit-def markers", multiline);

    const std::string comments = R"PTX(
// .shared .align 4 .u32 line_comment[2];
/* .local .align 8 .b8 block_comment[16]; */
".const .align 8 .b8 quoted_const[3];"
".global .align 16 .b8 quoted_global[5];"
"// implicit-def: %quoted"
.shared /* separator */ .align 4 .u32 interrupted[2];
.local // separator
.align 8 .b8 interrupted[8];
.const /* separator */ .align 8 .b8 interrupted[8];
.global // separator
.align 8 .b8 interrupted[8];
)PTX";
    ok &= compare("raw comments and strings retain legacy scanning behavior", comments);
    std::string noise;
    for (int i = 0; i < 256; ++i) noise += "add.u32 %r0, %r1, %r2;\n";
    ok &= compare("ordinary instruction text without candidates", noise);
    ok &= compare("declarations after ordinary instruction text", noise + declarations);
    const std::string padded = declarations + ".global .align 8 .b8 beyond_view[8];";
    ok &= compare("bounded string_view excludes trailing declaration",
                  std::string_view(padded).substr(0, declarations.size()));
    if (!ok) {
        std::cerr << "FAIL: module scan captures or decoded field values differ\n";
        return 1;
    }
    std::cout << "PASS: module scan legacy equivalence and decoded fields\n";
    return 0;
}
