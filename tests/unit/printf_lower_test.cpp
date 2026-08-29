#include "cumetal/passes/printf_lower.h"
#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool contains_warning(const std::vector<std::string>& warnings, const std::string& needle) {
    for (const auto& warning : warnings) {
        if (warning.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry k(
    .param .u64 p0
)
{
    call.uni (%r0), vprintf, ("tid=%d", %r1);
    call.uni (%r2), printf, ("tid=%d", %r2);
    call.uni (%r3), vprintf, ("%f", %f1);
    call.uni (%r4), foo, (%r4);
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(ptx);
    if (!expect(parsed.ok, "parse PTX for printf lowering")) {
        return 1;
    }

    const auto lowered = cumetal::passes::lower_printf_calls(parsed.module.entries[0]);
    if (!expect(lowered.ok, "printf lowering succeeds")) {
        return 1;
    }
    if (!expect(lowered.calls.size() == 3, "three printf/vprintf calls lowered")) {
        return 1;
    }
    if (!expect(lowered.formats.size() == 2, "deduplicated format table has two entries")) {
        return 1;
    }
    if (!expect(lowered.calls[0].format_id == lowered.calls[1].format_id,
                "duplicate format literal reuses format id")) {
        return 1;
    }
    if (!expect(lowered.calls[0].arguments.size() == 1 &&
                    lowered.calls[0].arguments[0] == "%r1",
                "printf arguments captured after format token")) {
        return 1;
    }
    if (!expect(lowered.formats[0].literal, "first format marked literal")) {
        return 1;
    }
    if (!expect(!lowered.formats[0].truncated, "first format not truncated")) {
        return 1;
    }

    const std::string long_format(300, 'a');
    const std::string trunc_ptx =
        ".version 8.0\n"
        ".target sm_90\n"
        ".visible .entry trunc(\n"
        "    .param .u64 p0\n"
        ")\n"
        "{\n"
        "    call.uni (%r0), vprintf, (\"" +
        long_format +
        "\", %r1);\n"
        "    ret;\n"
        "}\n";

    const auto trunc_parsed = cumetal::ptx::parse_ptx(trunc_ptx);
    if (!expect(trunc_parsed.ok, "parse PTX with long format literal")) {
        return 1;
    }

    const auto trunc_lowered = cumetal::passes::lower_printf_calls(trunc_parsed.module.entries[0]);
    if (!expect(trunc_lowered.ok, "printf lowering handles long format literal")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats.size() == 1, "single format in truncation test")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats[0].truncated, "long format literal truncated")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats[0].token.size() == 256, "format truncation obeys 256-byte limit")) {
        return 1;
    }
    if (!expect(contains_warning(trunc_lowered.warnings, "truncated"),
                "truncation warning emitted")) {
        return 1;
    }

    const std::string malformed_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry malformed(
    .param .u64 p0
)
{
    call.uni (%r0), vprintf;
    ret;
}
)PTX";

    const auto malformed_parsed = cumetal::ptx::parse_ptx(malformed_ptx);
    if (!expect(malformed_parsed.ok, "parse PTX with malformed printf call")) {
        return 1;
    }

    const auto tolerant = cumetal::passes::lower_printf_calls(malformed_parsed.module.entries[0]);
    if (!expect(tolerant.ok, "tolerant printf lowering keeps malformed call as warning")) {
        return 1;
    }
    if (!expect(contains_warning(tolerant.warnings, "missing argument tuple"),
                "missing argument tuple warning emitted")) {
        return 1;
    }

    cumetal::passes::PrintfLowerOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::passes::lower_printf_calls(malformed_parsed.module.entries[0], strict_options);
    if (!expect(!strict.ok, "strict printf lowering rejects malformed call")) {
        return 1;
    }
    if (!expect(strict.error.find("missing argument tuple") != std::string::npos,
                "strict error reports missing argument tuple")) {
        return 1;
    }

    const std::string clang_abi_ptx = R"PTX(
.version 7.0
.target sm_80
.global .align 1 .b8 _$_str[8] = {120, 61, 37, 100, 10, 0, 0, 0};
.visible .entry clang_printf(.param .u32 value) {
    .local .align 8 .b8 __local_depot0[16];
    mov.b64 %SPL, __local_depot0;
    cvta.local.u64 %SP, %SPL;
    add.u64 %rd5, %SPL, 0;
    st.local.b8 [%rd5], 32;
    add.u64 %rd1, %SP, 8;
    add.u64 %rd2, %SPL, 8;
    ld.param.b32 %r1, [value];
    st.local.v2.b32 [%rd2], {%r1, %r2};
    st.param.b64 [param1], %rd1;
    mov.b64 %rd3, _$_str;
    cvta.global.u64 %rd4, %rd3;
    st.param.b64 [param0], %rd4;
    call.uni (retval0), vprintf, (param0, param1);
    ret;
}
)PTX";
    const auto clang_parsed = cumetal::ptx::parse_ptx(clang_abi_ptx);
    if (!expect(clang_parsed.ok, "parse Clang vprintf ABI PTX")) return 1;
    cumetal::passes::PrintfLowerOptions clang_options;
    clang_options.ptx_source = clang_abi_ptx;
    const auto clang_lowered =
        cumetal::passes::lower_printf_calls(clang_parsed.module.entries[0], clang_options);
    if (!expect(clang_lowered.ok && clang_lowered.calls.size() == 1,
                "Clang vprintf ABI call lowers")) return 1;
    if (!expect(clang_lowered.formats.size() == 1 &&
                    clang_lowered.formats[0].token == "x=%d\n" &&
                    clang_lowered.formats[0].literal,
                "initialized global format bytes become a literal format")) return 1;
    if (!expect(clang_lowered.calls[0].arguments.size() == 2 &&
                    clang_lowered.calls[0].arguments[0] == "%r1" &&
                    clang_lowered.calls[0].arguments[1] == "%r2",
                "vector packed local argument tuple is decoded")) return 1;
    if (!expect(!clang_lowered.calls[0].abi_scaffold_lines.empty(),
                "Clang ABI pointer scaffolding is identified")) return 1;

    const std::string clang_no_args_ptx = R"PTX(
.version 7.0
.target sm_80
.global .align 1 .b8 _$_plain[12] = {114, 101, 97, 100, 121, 32, 49, 48, 48, 37, 37, 0};
.visible .entry clang_printf_no_args() {
    mov.b64 %rd1, _$_plain;
    cvta.global.u64 %rd2, %rd1;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], 0;
    call.uni (retval0), vprintf, (param0, param1);
    ret;
}
)PTX";
    const auto clang_no_args_parsed = cumetal::ptx::parse_ptx(clang_no_args_ptx);
    if (!expect(clang_no_args_parsed.ok, "parse null-tuple Clang vprintf ABI PTX")) return 1;
    clang_options.ptx_source = clang_no_args_ptx;
    const auto clang_no_args_lowered = cumetal::passes::lower_printf_calls(
        clang_no_args_parsed.module.entries[0], clang_options);
    if (!expect(clang_no_args_lowered.ok && clang_no_args_lowered.calls.size() == 1 &&
                    clang_no_args_lowered.calls[0].arguments.empty() &&
                    clang_no_args_lowered.formats[0].token == "ready 100%%",
                "Clang null vprintf tuple lowers when the format consumes no arguments")) {
        return 1;
    }

    std::string wide_clang_ptx = clang_abi_ptx;
    const std::size_t wide_store = wide_clang_ptx.find("st.local.v2.b32 [%rd2], {%r1, %r2};");
    if (!expect(wide_store != std::string::npos, "find Clang tuple store fixture")) return 1;
    wide_clang_ptx.replace(wide_store,
                           std::string("st.local.v2.b32 [%rd2], {%r1, %r2};").size(),
                           "st.local.b64 [%rd2], %rd5;");
    const auto wide_parsed = cumetal::ptx::parse_ptx(wide_clang_ptx);
    if (!expect(wide_parsed.ok, "parse 64-bit Clang tuple")) return 1;
    clang_options.ptx_source = wide_clang_ptx;
    const auto wide_lowered =
        cumetal::passes::lower_printf_calls(wide_parsed.module.entries[0], clang_options);
    if (!expect(wide_lowered.ok && wide_lowered.formats.size() == 1 &&
                    wide_lowered.formats[0].literal &&
                    wide_lowered.calls.size() == 1 &&
                    wide_lowered.calls[0].arguments.size() == 1 &&
                    wide_lowered.calls[0].arguments[0] == "%rd5" &&
                    wide_lowered.calls[0].argument_bits.size() == 1 &&
                    wide_lowered.calls[0].argument_bits[0] == 64,
                "64-bit Clang tuple is preserved as two runtime words")) return 1;

    std::printf("PASS: printf lower unit tests\n");
    return 0;
}
