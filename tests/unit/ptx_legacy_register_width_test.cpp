#include "cumetal/ptx/lower_to_llvm.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Case {
    std::string name;
    std::string body;
    std::vector<std::pair<std::string, int>> slots;
    std::string required_ir = {};
    std::string error = {};
};

std::string kernel(const std::string& body) {
    return R"PTX(.version 7.0
.target sm_70
.address_size 64
.visible .entry contract(.param .u64 destination) {
    .reg .u64 %output;
    ld.param.u64 %output, [destination];
)PTX" + body +
           "\n    ret;\n}\n";
}

bool check(const Case& test, const std::filesystem::path& output) {
    cumetal::ptx::LowerToLlvmOptions options;
    options.strict = true;
    options.entry_name = "contract";
    const auto result = cumetal::ptx::lower_ptx_to_llvm_ir(kernel(test.body), options);
    if (!test.error.empty()) {
        if (result.ok || !result.llvm_ir.empty() || result.error.find(test.error) == std::string::npos) {
            std::fprintf(stderr, "FAIL %s: expected rejection containing '%s', got ok=%d '%s'\n",
                         test.name.c_str(), test.error.c_str(), result.ok, result.error.c_str());
            return false;
        }
        return true;
    }
    if (!result.ok) {
        std::fprintf(stderr, "FAIL %s: %s\n", test.name.c_str(), result.error.c_str());
        return false;
    }
    for (const auto& [name, width] : test.slots) {
        const std::regex allocation("%cm_reg_" + name + "_[0-9]+ = alloca i" + std::to_string(width) + ",");
        if (!std::regex_search(result.llvm_ir, allocation)) {
            std::fprintf(stderr, "FAIL %s: missing i%d slot for %s\n", test.name.c_str(), width,
                         name.c_str());
            return false;
        }
    }
    if (!test.required_ir.empty() && result.llvm_ir.find(test.required_ir) == std::string::npos) {
        std::fprintf(stderr, "FAIL %s: missing operation '%s'\n", test.name.c_str(),
                     test.required_ir.c_str());
        return false;
    }
    // The companion compiler/assembler regression verifies every emitted file,
    // in addition to the complete six-fixture joined ReLU denominator.
    if (!output.empty()) {
        std::ofstream file(output / (test.name + ".ll"));
        file << result.llvm_ir;
        if (!file)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [LLVM_OUTPUT_DIRECTORY]\n", argv[0]);
        return 1;
    }
    const std::filesystem::path output = argc == 2 ? argv[1] : "";
    if (!output.empty())
        std::filesystem::create_directories(output);
    const std::string maximum = std::to_string(std::numeric_limits<std::size_t>::max());
    const std::string last = std::to_string(std::numeric_limits<std::size_t>::max() - 1);
    const std::vector<Case> cases = {
        {"explicit_pointer_copy",
         R"PTX(
    .reg .b64 %SP, %SPL;
    ld.param.u64 %SP, [destination];
    mov.b64 %SPL, %SP;
    add.u64 %SPL, %SPL, 8;
    st.global.u32 [%SPL], 7;
)PTX",
         {{"SP", 64}, {"SPL", 64}}},
        {"arbitrary_range",
         R"PTX(
    .reg .u64 %address<4>;
    .reg .u32 %index;
    mov.u32 %index, 4;
    mov.b64 %address0, %output;
    mov.b64 %address1, %address0;
    cvt.u64.u32 %address2, %index;
    add.u64 %address3, %address1, %address2;
    st.global.u32 [%address3], 7;
)PTX",
         {{"address0", 64}, {"address1", 64}, {"address2", 64}, {"address3", 64}}},
        {"misleading_spelling",
         R"PTX(
    .reg .u32 %rd_value;
    .reg .u64 %r_wide;
    .reg .pred %condition;
    mov.u32 %rd_value, 17;
    mov.u64 %r_wide, 4294967297;
    setp.eq.u32 %condition, %rd_value, 0;
    @%condition bra DONE;
    st.global.u64 [%output], %r_wide;
DONE:
)PTX",
         {{"rd_value", 32}, {"r_wide", 64}, {"condition", 1}}},
        {"conversion_storage",
         R"PTX(
    .reg .b16 %small;
    .reg .b32 %value;
    .reg .b64 %wide;
    mov.b16 %small, 65535;
    cvt.s32.s16 %value, %small;
    cvt.u64.u32 %wide, %value;
    st.global.u64 [%output], %wide;
)PTX",
         {{"small", 16}, {"value", 32}, {"wide", 64}},
         "sext i16"},
        {"signed_narrow_load",
         R"PTX(
    .reg .b32 %value;
    ld.global.s8 %value, [%output];
    st.global.b32 [%output], %value;
)PTX",
         {{"value", 32}},
         "sext i8"},
        {"unsigned_narrow_load",
         R"PTX(
    .reg .b32 %value;
    ld.global.u8 %value, [%output];
    st.global.b32 [%output], %value;
)PTX",
         {{"value", 32}},
         "zext i8"},
        {"wide_multiply",
         R"PTX(
    .reg .s16 %left, %right;
    .reg .b32 %product;
    mov.s16 %left, 7;
    mov.s16 %right, -3;
    mul.wide.s16 %product, %left, %right;
    st.global.b32 [%output], %product;
)PTX",
         {{"left", 16}, {"right", 16}, {"product", 32}},
         "mul i32"},
        {"tuple_storage",
         R"PTX(
    .reg .b16 %low, %high;
    .reg .b32 %packed;
    mov.b16 %low, 1;
    mov.b16 %high, 2;
    mov.b32 %packed, {%low, %high};
    mov.b32 {%low, %high}, %packed;
    st.global.b32 [%output], %packed;
)PTX",
         {{"low", 16}, {"high", 16}, {"packed", 32}}},
        {"unique_scoped",
         R"PTX(
    {
        .reg .u64 %temporary;
        mov.u64 %temporary, 4294967297;
        st.global.u64 [%output], %temporary;
    }
)PTX",
         {{"temporary", 64}}},
        {"undeclared_compatibility",
         R"PTX(
    mov.u64 %rd_compat, 4294967297;
    st.global.u64 [%output], %rd_compat;
)PTX",
         {{"rd_compat", 64}}},
        {"compact_huge_range",
         ".reg .u64 %large<" + maximum +
             ">;\n"
             "mov.u64 %large0, 4294967297;\n"
             "mov.u64 %large" +
             last +
             ", %large0;\n"
             "st.global.u64 [%output], %large" +
             last + ";\n",
         {{"large0", 64}, {"large" + last, 64}}},
        // Out-of-range/noncanonical names retain the preexisting undeclared
        // fallback; they must not accidentally inherit a range's u64 contract.
        {"range_boundaries",
         R"PTX(
    .reg .u64 %address<2>;
    mov.u32 %address2, 3;
    mov.u32 %address01, 4;
    mov.u32 %address184467440737095516160, 5;
    st.global.u32 [%output], %address2;
)PTX",
         {{"address2", 32}, {"address01", 32}, {"address184467440737095516160", 32}}},
        {"exact_conflict",
         R"PTX(
    .reg .u64 %value;
    .reg .u32 %value;
    mov.u32 %value, 1;
)PTX",
         {},
         {},
         "conflicting legacy LLVM declared register widths"},
        {"exact_range_conflict",
         R"PTX(
    .reg .u64 %value<2>;
    .reg .u32 %value1;
    mov.u32 %value1, 1;
)PTX",
         {},
         {},
         "conflicting legacy LLVM declared register widths"},
        {"overlapping_ranges",
         R"PTX(
    .reg .u64 %value<2>;
    .reg .u32 %value<3>;
    mov.u32 %value1, 1;
)PTX",
         {},
         {},
         "conflicting legacy LLVM declared register widths"},
        {"different_prefix_overlap",
         R"PTX(
    .reg .u64 %value<20>;
    .reg .u32 %value1<2>;
    mov.u32 %value10, 1;
)PTX",
         {},
         {},
         "conflicting legacy LLVM declared register widths"},
        {"scoped_width_conflict",
         R"PTX(
    .reg .u64 %value;
    mov.u64 %value, 4294967297;
    {
        .reg .u32 %value;
        mov.u32 %value, 1;
    }
)PTX",
         {},
         {},
         "conflicting legacy LLVM declared register widths"},
        {"scoped_same_width_alias",
         R"PTX(
    .reg .u64 %value;
    mov.u64 %value, 4294967297;
    {
        .reg .u64 %value;
        mov.u64 %value, 1;
    }
    st.global.u64 [%output], %value;
)PTX",
         {},
         {},
         "ambiguous scoped legacy LLVM register declarations"},
        {"unknown_declaration_type",
         R"PTX(
    .reg .b64junk %value;
    mov.u64 %value, 1;
)PTX",
         {},
         {},
         "unsupported legacy LLVM declared register type"},
        {"packed_declaration_not_scalar16",
         R"PTX(
    .reg .f16x2 %value;
    mov.b32 %value, 1;
)PTX",
         {},
         {},
         "unsupported legacy LLVM declared register type"},
        {"raw_address_width_rejected",
         R"PTX(
    .reg .u32 %address, %value;
    mov.u32 %address, 0;
    ld.global.u32 %value, [%address];
    st.global.u32 [%output], %value;
)PTX",
         {},
         {},
         "raw register load width mismatch"},
        {"raw_predicate_width_rejected",
         R"PTX(
    .reg .u32 %condition;
    mov.u32 %condition, 1;
    @%condition bra DONE;
    st.global.u32 [%output], 7;
DONE:
)PTX",
         {},
         {},
         "raw register load width mismatch"},
    };
    bool ok = true;
    for (const auto& test : cases)
        ok = check(test, output) && ok;
    if (ok)
        std::printf("PASS: %zu legacy register storage contracts\n", cases.size());
    return ok ? 0 : 1;
}
