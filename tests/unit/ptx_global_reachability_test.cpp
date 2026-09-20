#include "cumetal/metal/lower_to_msl.h"
#include "ptx_module.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
using cumetal::metal::compile_ptx_to_msl;
bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}
const std::string header = ".version 7.0\n.target sm_80\n.address_size 64\n";
const std::string unused_alias =
    ".global .align 8 .u8 bad[8] = {0xFF(target),0,0,0,0,0,0,0};\n";
const std::string numeric_table =
    ".global .align 4 .b8 bad_suffix[4] = {42,0,0,0};\n";
const std::string good_entry = R"ptx(
.visible .entry good(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [bad_suffix];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
}

int main() {
    bool ok = true;
    // Storage classification must follow aliases, independently of address conversions.
    for (const std::string name : {"table", "__const_$literal"}) {
        const std::string prefix = header + ".global .align 4 .b8 " + name + R"ptx([4] = {1,0,0,0};
.visible .entry alias_write() {
.reg .b64 %rd<3>;
.reg .b32 %r1;
.reg .pred %p1;
)ptx" + "mov.u64 %rd1, " + name + ";\n";
        for (const std::string write : {
                 "st.global.u8 [%rd1], 7;",
                 "mov.u64 %rd2, %rd1;\nst.global.u8 [%rd2], 7;",
                 "setp.eq.u32 %p1, 0, 0;\n@%p1 st.global.u8 [%rd1], 7;",
                 "mov.u64 %rd2, %rd1;\nmov.u64 %rd1, 0;\nst.global.u8 [%rd2], 7;",
                 "atom.global.add.u32 %r1, [%rd1], 1;",
                 "red.global.add.u32 [%rd1], 1;"}) {
            const auto source = prefix + write + "\nret;\n}\n";
            const auto parsed = cumetal::ptx::parse_ptx(source);
            ok &= expect(parsed.ok && cumetal::ir::detail::symbol_is_written(parsed.module, name),
                         "alias analysis observes each write form for " + name);
            // These forms are parsed but have separate lowering limitations.
            if (write.find("@%p1") != std::string::npos || write.starts_with("red.")) continue;
            const auto written = compile_ptx_to_msl(source);
            ok &= expect(written.ok && written.gpu_ir.global_constants.empty(),
                         "aliased writes retain mutable storage for " + name + ": " + written.error);
        }
        const auto read_only = compile_ptx_to_msl(prefix + "ld.global.u8 %r1, [%rd1];\nret;\n}\n");
        ok &= expect(read_only.ok && read_only.gpu_ir.global_constants.size() == 1,
                     "read-only aliases still permit promotion for " + name + ": " + read_only.error);
    }

    cumetal::metal::PtxToMslOptions options;
    options.strict = true;
    options.entry_name = "good";
    const std::string unsupported_entry =
        ".visible .entry unsupported() {\nunsupported_opcode;\nret;\n}\n";
    auto selected = compile_ptx_to_msl(header + numeric_table + good_entry + unsupported_entry, options);
    ok &= expect(selected.ok, "unselected unsupported instructions do not reject the selected entry");
    options.entry_name = "unsupported";
    selected = compile_ptx_to_msl(header + numeric_table + good_entry + unsupported_entry, options);
    ok &= expect(!selected.ok && selected.error.find("unsupported opcode") != std::string::npos,
                 "strict opcode validation still rejects selected unsupported instructions");
    options.entry_name = "good";
    const std::string other_users = R"ptx(
.func unused_helper() {
    .reg .b64 %rd1;
    mov.u64 %rd1, bad;
    ret;
}
.visible .entry other() {
    call unused_helper, ();
    ret;
}
)ptx";
    auto result = compile_ptx_to_msl(header + unused_alias + numeric_table +
                                   good_entry + other_users, options);
    ok &= expect(result.ok, "unused alias and unreachable helper do not block good: " + result.error);
    ok &= expect(result.gpu_ir.global_constants.size() == 1 &&
                 result.gpu_ir.global_constants.front().name == "bad_suffix" &&
                 result.gpu_ir.global_constants.front().bytes == std::vector<std::uint8_t>({42, 0, 0, 0}),
                 "selected initialized data retains its exact bytes; symbol prefixes do not match");

    options.entry_name = "other";
    result = compile_ptx_to_msl(header + unused_alias + numeric_table + good_entry + other_users, options);
    ok &= expect(!result.ok && result.error.find("initialized PTX byte array") != std::string::npos,
                 "the same alias fails when its helper becomes reachable");

    options.entry_name = "root";
    const std::string root = ".visible .entry root() {\ncall middle, ();\nret;\n}\n";
    const std::string middle = ".func middle() {\ncall leaf, ();\nret;\n}\n";
    for (const auto* operand : {"bad", "bad+1", "generic(bad)", "[bad+1]", "[bad-1]"}) {
        const std::string leaf = ".func leaf() {\n.reg .b64 %rd1;\nmov.u64 %rd1, " +
                                 std::string(operand) + ";\nret;\n}\n";
        result = compile_ptx_to_msl(header + unused_alias + root + middle + leaf, options);
        ok &= expect(!result.ok && result.error.find("initialized PTX byte array") != std::string::npos,
                     "transitive helper reference retains initializer validation: " + std::string(operand));
    }

    // Incomplete address expressions must never become zero data.
    const std::string aliases =
        ".global .align 8 .b8 bad[8] = {0xFF(second),0,0,0,0,0,0,0};\n"
        ".global .align 8 .b8 second[8] = {0xFF(bad),0,0,0,0,0,0,0};\n";
    const std::string direct = ".visible .entry root() {\n.reg .b64 %rd1;\nmov.u64 %rd1, bad;\nret;\n}\n";
    result = compile_ptx_to_msl(header + aliases + direct, options);
    ok &= expect(!result.ok && result.error.find("initialized PTX byte array") != std::string::npos,
                 "a reached alias cycle is explicitly rejected, not dropped or evaluated");
    result = compile_ptx_to_msl(header + aliases + ".visible .entry root() {\nret;\n}\n", options);
    ok &= expect(result.ok, "an entirely unused alias cycle is omitted: " + result.error);

    for (const auto* initializer : {"{1,2,3,4,5}", "{999}", "{0xFF(second)}"}) {
        result = compile_ptx_to_msl(header + ".global .align 4 .b8 bad[4] = " +
                                   initializer + ";\n" + direct, options);
        ok &= expect(!result.ok && result.error.find("initialized PTX byte array") != std::string::npos,
                     "reached malformed or unsupported byte initializer still fails");
    }
    result = compile_ptx_to_msl(header + ".global .align 8 .u8 [8] = {0};\n" +
                               ".visible .entry root() {\nret;\n}\n", options);
    ok &= expect(!result.ok && result.error.find("cannot identify initialized PTX declaration") != std::string::npos,
                 "unidentifiable declaration cannot be silently filtered");

    // A numeric operand must not accidentally match a symbol inside the token.
    result = compile_ptx_to_msl(header + ".global .align 8 .u8 xFF[8] = {0};\n" +
        ".visible .entry root() {\n.reg .b32 %r1;\nmov.u32 %r1, 0xFF;\nret;\n}\n", options);
    ok &= expect(result.ok, "hex literal does not make xFF reachable: " + result.error);

    // Selection must not turn module-wide mutable storage into a constant.
    options.entry_name = "good";
    result = compile_ptx_to_msl(header + numeric_table + good_entry +
        ".visible .entry writer() {\nst.global.b32 [bad_suffix], 7;\nret;\n}\n", options);
    ok &= expect(result.ok && result.source.find("device uchar* cm___cumetal_global_bad_suffix") != std::string::npos &&
                 result.source.find("constant uchar cm_bad_suffix") == std::string::npos,
                 "writes in another entry preserve shared mutable global storage");

    options.entry_name = "relocated";
    const auto alias_to = [](const std::string& name, const std::string& target) {
        std::string value = ".global .align 8 .u8 " + name + "[8] = {";
        for (int i = 0; i < 8; ++i) {
            if (i) value += ",";
            value += "0XFF" + std::string(i * 2, '0') + "(" + target + ")";
        }
        return value + "};\n";
    };
    const std::string table = ".global .align 4 .b8 table[4] = {42,17,99,5};\n";
    const std::string relocated_entry = R"ptx(
.visible .entry relocated(.param .u64 output) {
    .reg .b64 %rd<3>;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.global.nc.u64 %rd2, [alias];
    ld.global.u32 %r1, [%rd2];
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";
    result = compile_ptx_to_msl(".version 7.1\n.target sm_80\n.address_size 32\n// .address_size 64\n" +
        table + alias_to("alias", "table") + relocated_entry, options);
    ok &= expect(!result.ok && result.error.find("64-bit addressing") != std::string::npos,
                 "32-bit modules cannot use the 64-bit relocation fold");
    for (const bool forward : {false, true}) {
        result = compile_ptx_to_msl(header +
            (forward ? alias_to("alias", "table") + table : table + alias_to("alias", "table")) +
            relocated_entry, options);
        ok &= expect(result.ok, "complete relocation resolves regardless of declaration order: " + result.error);
        ok &= expect(result.gpu_ir.global_constants.size() == 1 &&
                     result.gpu_ir.global_constants.front().name == "table" &&
                     result.gpu_ir.global_constants.front().bytes == std::vector<std::uint8_t>({42,17,99,5}),
                     "relocation retains dependency data without emitting placeholder pointer bytes");
    }

    const std::string helper = R"ptx(
.func read_table() {
 .reg .b64 %rd1;
 .reg .b32 %r1;
 ld.global.nc.u64 %rd1, [alias];
 ld.global.u32 %r1, [%rd1];
 ret;
}
.visible .entry relocated() {
 call read_table, ();
 ret;
}
)ptx";
    result = compile_ptx_to_msl(header + table + alias_to("alias", "table") + helper, options);
    ok &= expect(result.ok && result.gpu_ir.global_constants.size() == 1,
                 "relocation referenced solely by a reachable helper retains its table: " + result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {".global", ".visible .global"}, {".align 8", ".align 4"},
        {".align 8", ".align 12"}, {"[8]", "[9]"},
        {"0XFF(table)", "0XFFFFFFFFFFFFFFFFFF(table)"},
        {"0XFF(table)", "0XFE(table)"}}) {
        std::string declaration = alias_to("alias", "table");
        declaration.replace(declaration.find(from), from.size(), to);
        result = compile_ptx_to_msl(header + table + declaration + relocated_entry, options);
        ok &= expect(!result.ok, "exported, misaligned, wrong-size and invalid-mask relocations reject: " + to);
    }
    result = compile_ptx_to_msl(header + table + table + alias_to("alias", "table") + relocated_entry, options);
    ok &= expect(!result.ok && result.error.find("duplicate initialized PTX symbol") != std::string::npos,
                 "duplicate initialized declarations reject instead of picking an arbitrary target");
    for (const std::string extra : {
        ".visible .entry writer() {\nst.global.u64 [alias], 0;\nret;\n}\n",
        ".visible .entry escape() {\n.reg .b64 %rd1;\nmov.u64 %rd1, alias;\nret;\n}\n",
        ".visible .entry partial() {\n.reg .b32 %r1;\nld.global.u32 %r1, [alias];\nret;\n}\n",
        ".visible .entry displaced() {\n.reg .b64 %rd1;\nld.global.u64 %rd1, [alias+1];\nret;\n}\n",
        ".visible .entry indirect() {\n.reg .b64 %rd1;\nmov.u64 %rd1, table;\nst.global.u32 [%rd1], 0;\nret;\n}\n",
        ".visible .entry indirect() {\n.reg .b64 %rd1;\nld.global.u64 %rd1, [alias];\nst.global.u32 [%rd1], 0;\nret;\n}\n",
        ".global .align 8 .u64 another_pointer = table;\n",
        ".visible .entry writer() {\nst.global.u32 [table], 0;\nret;\n}\n"}) {
        result = compile_ptx_to_msl(header + table + alias_to("alias", "table") + relocated_entry + extra, options);
        ok &= expect(!result.ok && result.error.find("PTX") != std::string::npos,
                     "mutable, escaping, partial and offset relocation uses fail explicitly");
    }
    for (const std::string target : {"missing", "alias", "second"}) {
        result = compile_ptx_to_msl(header + alias_to("alias", target) +
            alias_to("second", "alias") + relocated_entry, options);
        ok &= expect(!result.ok, "unresolved and cyclic relocation targets are rejected");
    }
    for (const std::string bytes : {
        "0xFF(table),0xFF00(table)",
        "0xFF00(table),0xFF(table),0,0,0,0,0,0",
        "0xFF(table),0xFF00(other),0,0,0,0,0,0",
        "0,0xFF00(table),0,0,0,0,0,0"}) {
        result = compile_ptx_to_msl(header + table + ".global .align 8 .u8 alias[8] = {" +
            bytes + "};\n" + relocated_entry, options);
        ok &= expect(!result.ok, "incomplete, reordered, mixed-target and mixed-numeric relocations reject");
    }
    // Typed pointer initializers must use the same guarded relocation path.
    for (const std::string type : {"u64", "b64"}) {
        const std::string alias = ".global .align 8 ." + type + " alias[1] = {table};\n";
        for (const bool forward : {false, true}) {
            result = compile_ptx_to_msl(header + (forward ? alias + table : table + alias) + relocated_entry, options);
            ok &= expect(result.ok && result.gpu_ir.global_constants.size() == 1 &&
                         result.gpu_ir.global_constants.front().bytes == std::vector<std::uint8_t>({42,17,99,5}),
                         "typed pointer retains only the target bytes: " + result.error);
        }
        result = compile_ptx_to_msl(header + table + alias + helper, options);
        ok &= expect(result.ok, "typed relocation works through reachable helpers: " + result.error);
        for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
            {".global", ".visible .global"}, {".align 8", ".align 4"},
            {".align 8", ".align 12"}, {"[1]", "[2]"},
            {"{table}", "{table+8}"}, {"{table}", "{table,table}"},
            {"{table}", "{missing}"}, {"{table}", "{alias}"}}) {
            std::string changed = alias;
            changed.replace(changed.find(from), from.size(), to);
            result = compile_ptx_to_msl(header + table + changed + relocated_entry, options);
            ok &= expect(!result.ok, "invalid typed pointer rejects: " + to);
        }
        for (const std::string extra : {
            ".visible .entry writer() {\nst.global.u64 [alias], 0;\nret;\n}\n",
            ".visible .entry writer() {\nst.global.u32 [table], 0;\nret;\n}\n",
            ".visible .entry partial() {\n.reg .b32 %r1;\nld.global.u32 %r1, [alias];\nret;\n}\n",
            ".visible .entry escape() {\n.reg .b64 %rd1;\nmov.u64 %rd1, alias;\nret;\n}\n"}) {
            result = compile_ptx_to_msl(header + table + alias + relocated_entry + extra, options);
            ok &= expect(!result.ok, "typed pointer retains mutation, partial-load and escape checks");
        }
        result = compile_ptx_to_msl(".version 7.1\n.target sm_80\n.address_size 32\n" + table + alias + relocated_entry, options);
        ok &= expect(!result.ok, "typed relocation requires 64-bit addressing");
    }
    return ok ? 0 : 1;
}
