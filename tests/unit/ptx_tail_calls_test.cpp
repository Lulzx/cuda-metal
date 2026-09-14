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
