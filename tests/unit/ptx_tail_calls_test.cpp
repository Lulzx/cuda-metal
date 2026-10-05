#include "cumetal/metal/lower_to_msl.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto fixture = [&](const char* name) {
        std::ifstream input(std::filesystem::path(argv[1]) / name);
        if (!input) throw std::runtime_error(std::string("missing fixture: ") + name);
        return std::string(std::istreambuf_iterator<char>(input), {});
    };
    namespace metal = cumetal::metal;
    bool ok = true;
    const std::string wide_return = [&] {
        auto source = fixture("ptx_scalar_tail_call.ptx");
        const auto begin = source.find(".func"), end = source.find(".visible .entry");
        source.replace(begin, end - begin,
            ".func (.param .align 8 .b8 retval[16]) tail_count(.param .b64 seed) {\n"
            "st.param.b64 [retval], 1;\nst.param.b64 [retval+8], 2;\nret;\n}\n");
        return source;
    }();
    const auto wide_result = metal::compile_ptx_to_msl(wide_return);
    ok &= expect(wide_result.ok, "64-bit immediate stores populate byte-array returns: " + wide_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"[retval+8]", "[retval+4]"},
        {"[retval+8]", "[retval+16]"},
        {"[retval+8]", "[retval+unknown]"},
        {"[retval+8]", "[retval+8junk]"},
        {"[result+8]", "[result+unknown]"},
        {"[result+8]", "[result+8junk]"},
        {"[result+8]", "[result+4]"},
        {"[result+8]", "[result+16]"}}) {
        std::string invalid = wide_return;
        invalid.replace(invalid.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "malformed, overlapping, missing and out-of-bounds wide return fields rejected");
    }
    std::string vector_return = wide_return;
    for (std::size_t pos = 0; (pos = vector_return.find(".align 8 .b8", pos)) != std::string::npos; pos += 13)
        vector_return.replace(pos, 12, ".align 16 .b8");
    const std::string scalar_stores = "st.param.b64 [retval], 1;\nst.param.b64 [retval+8], 2;";
    vector_return.replace(vector_return.find(scalar_stores), scalar_stores.size(),
                          "st.param.v2.b64 [retval], {1, 2};");
    const std::string scalar_loads = "ld.param.b64 %rd8, [result];\nld.param.b64 %rd9, [result+8];";
    vector_return.replace(vector_return.find(scalar_loads), scalar_loads.size(),
                          "ld.param.v2.b64 {%rd8, %rd9}, [result];");
    const auto vector_result = metal::compile_ptx_to_msl(vector_return);
    ok &= expect(vector_result.ok, "both vector parameter lanes participate in aggregate ABI: " + vector_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"{1, 2}", "{1}"}, {"{1, 2}", "{1, unknown}"},
        {"{%rd8, %rd9}", "{%rd8, %rd8}"},
        {"{%rd8, %rd9}", "{%rd8, %r1}"},
        {"[result];", "[result+8];"},
        {"[result];", "[result+unknown];"},
        {"[result];", "[%rd8];"},
        {"st.param.v2.b64", "@%p1 st.param.v2.b64"}}) {
        std::string invalid = vector_return;
        invalid.replace(invalid.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "malformed, misaligned and predicated vector parameter transfers rejected");
    }
    for (unsigned width : {2u, 4u}) {
        const std::string tuple = width == 2 ? "{%r1, %r2}" : "{%r1, %r2, %r3, %r4}";
        const std::string opcode = "ld.param.v" + std::to_string(width) + ".b32";
        std::string source = ".version 7.0\n.target sm_80\n.address_size 64\n"
            ".visible .entry vector_input(.param .align 16 .b8 input[16], .param .u64 output) {\n"
            ".reg .b32 %r<5>;\n.reg .b64 %rd1;\nld.param.u64 %rd1, [output];\n" +
            opcode + " " + tuple + ", [input];\n";
        for (unsigned i = 0; i < width; ++i)
            source += "st.global.b32 [%rd1+" + std::to_string(4*i) + "], %r" + std::to_string(i+1) + ";\n";
        source += "ret;\n}\n";
        const auto result = metal::compile_ptx_to_msl(source);
        ok &= expect(result.ok, "32-bit vector parameter lanes preserve aggregate fields: " + result.error);
        auto indirect = source;
        indirect.replace(indirect.find(".reg .b64 %rd1;"), 15, ".reg .b64 %rd<3>;");
        indirect.insert(indirect.find(opcode), "mov.b64 %rd2, input;\n");
        indirect.replace(indirect.find("[input];"), 8, "[%rd2];");
        const auto indirect_result = metal::compile_ptx_to_msl(indirect);
        ok &= expect(indirect_result.ok, "indirect parameter vector lanes retain bounds and provenance: " + indirect_result.error);
        for (const auto& [from,to] : std::vector<std::pair<std::string,std::string>>{
            {tuple, "{%r1, %r1}"}, {"[input];", "[input+4];"},
            {opcode, "@%p1 " + opcode}}) {
            auto invalid = source;
            invalid.replace(invalid.find(from), from.size(), to);
            ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                "duplicate, misaligned, and predicated vector parameter lanes fail");
        }
    }
    const std::string packed_wide = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry packed_wide(.param .align 8 .b8 input[64], .param .u64 output) {
.reg .b32 %r<3>;
.reg .b64 %rd<3>;
ld.param.u64 %rd1, [output];
ld.param.v2.b32 {%r1,%r2}, [input+48];
ld.param.b64 %rd2, [input+8];
st.global.b32 [%rd1], %r1;
st.global.b32 [%rd1+4], %r2;
st.global.b64 [%rd1+8], %rd2;
ret;
})ptx";
    const auto packed_result = metal::compile_ptx_to_msl(packed_wide);
    ok &= expect(packed_result.ok, "Kokkos mixed 32/64-bit parameter record loads: " + packed_result.error);
    for (const auto& offset : {"+4]", "+60]", "+unknown]"}) {
        auto invalid = packed_wide;
        invalid.replace(invalid.find("+8]"), 3, offset);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "unaligned, partial, and malformed wide parameter loads fail");
    }
    const std::string cta_reduce = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry cta_reduce(.param .u64 output) {
.reg .pred %p<3>;
.reg .b32 %r1;
.reg .b64 %rd1;
ld.param.u64 %rd1,[output];
mov.pred %p1,1;
bar.red.or.pred %p2,0,%p1;
selp.u32 %r1,1,0,%p2;
st.global.u32 [%rd1],%r1;
ret;
})ptx";
    const auto cta_result = metal::compile_ptx_to_msl(cta_reduce);
    ok &= expect(cta_result.ok && cta_result.source.find("cm_cta_any") != std::string::npos,
                 "CTA reduction defines its predicate result: " + cta_result.error);
    for (const auto& replacement : {"bar.red.and.pred %p2,0,%p1;", "bar.red.or.pred %p2,1,%p1;",
                                    "bar.red.or.pred %p2,0,32,%p1;", "@%p1 bar.red.or.pred %p2,0,%p1;"}) {
        auto invalid = cta_reduce;
        const std::string original = "bar.red.or.pred %p2,0,%p1;";
        invalid.replace(invalid.find(original), original.size(), replacement);
        ok &= expect(!metal::compile_ptx_to_msl(invalid).ok,
                     "unsupported reduction operation, barrier ID, count, and predicate fail");
    }
    const std::string scalar_tail = fixture("ptx_scalar_tail_call.ptx");
    const auto tail_result = metal::compile_ptx_to_msl(scalar_tail);
    ok &= expect(tail_result.ok, "scalar aggregate-return tail call becomes a loop: " + tail_result.error);
    auto collision = scalar_tail;
    collision.insert(collision.find(".reg .pred"), ".reg .b64 %rd_cm_tail_argument;\n");
    collision.replace(collision.find("BASE:"), 5, "$cm_tail_header:\nBASE:");
    ok &= expect(metal::compile_ptx_to_msl(collision).ok, "tail rewrite avoids existing register and label names");
    auto hex_offset = scalar_tail;
    for (std::size_t pos = 0; (pos = hex_offset.find("+8]", pos)) != std::string::npos; pos += 5)
        hex_offset.replace(pos, 3, "+0x8]");
    ok &= expect(metal::compile_ptx_to_msl(hex_offset).ok, "tail and aggregate ABI use the same literal offset parser");
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"RETURN:\nst.param.b64 [retval], %rd3;", "RETURN:\nadd.u64 %rd3, %rd3, 1;\nst.param.b64 [retval], %rd3;"},
        {"ld.param.b64 %rd4, [result+8];", "ld.param.b64 %rd4, [result];"},
        {"st.param.b64 [retval+8], %rd4;", "st.param.b64 [retval+8], %rd3;"},
        {"call.uni (result), tail_count, (arg);", "@%p1 call.uni (result), tail_count, (arg);"},
        {"sub.u64 %rd2, %rd1, 1;", "ld.local.u64 %rd2, [%rd1];"},
        {"sub.u64 %rd2, %rd1, 1;", "add.u64 %rd2, depot, 0;"},
        {"st.param.b64 [arg], %rd2;", "st.param.b64 [arg+unknown], %rd2;"}}) {
        std::string unsupported = scalar_tail;
        unsupported.replace(unsupported.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(unsupported);
        ok &= expect(!rejected.ok && rejected.error.find("recursive PTX") != std::string::npos,
                     "non-tail, malformed, predicated and memory-dependent recursion remain rejected: " + rejected.error);
    }
    return ok ? 0 : 1;
}
