#include "cumetal/ir/ptx_importer.h"
#include "trap_call_expansion.h"
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
    const std::string trap_probe = fixture("ptx_trap_reporting.ptx");
    const auto trap_result = metal::compile_ptx_to_msl(trap_probe);
    ok &= expect(trap_result.ok && trap_result.source.find("atomic_fetch_or_explicit(cm_trap_status") != std::string::npos &&
                     trap_result.source.find("atomic_load_explicit(cm_trap_status") != std::string::npos,
                 "kernel traps report failure and poll cancellation: " + trap_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"FAULT:\ntrap;", "FAULT:\nbar.sync 0;\ntrap;"},
        {"output", "cm_trap_status"}}) {
        std::string invalid = trap_probe;
        for (std::size_t at = 0; (at = invalid.find(from, at)) != std::string::npos; at += to.size())
            invalid.replace(at, from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "trap barriers and hidden argument collisions rejected");
    }
    const auto helper_trap = metal::compile_ptx_to_msl(R"ptx(
.version 7.1
.target sm_80
.func helper() {
trap;
}
.visible .entry helper_trap() {
call.uni helper, ();
ret;
}
)ptx");
    ok &= expect(helper_trap.ok && helper_trap.source.find("atomic_fetch_or_explicit(cm_trap_status") != std::string::npos,
                 "helper trap expands into kernel cancellation CFG: " + helper_trap.error);
    const std::string nested_barrier = R"ptx(
.version 7.1
.target sm_80
.func barrier_helper() {
bar.sync 0;
ret;
}
.func caller() {
call.uni barrier_helper, ();
trap;
}
.visible .entry nested_barrier() {
call.uni caller, ();
ret;
}
)ptx";
    const auto rejected_barrier = metal::compile_ptx_to_msl(nested_barrier);
    ok &= expect(!rejected_barrier.ok && rejected_barrier.error.find("without barriers or collectives") != std::string::npos,
                 "barriers in expanded call graphs remain rejected: " + rejected_barrier.error);
    std::string oversized = R"ptx(
.version 7.1
.target sm_80
.func noop(.param .u32 flag) {
.reg .b32 %r1;
.reg .pred %p1;
ld.param.u32 %r1, [flag];
setp.eq.u32 %p1, %r1, 1;
@%p1 bra FAIL;
ret;
FAIL:
trap;
}
.visible .entry oversized() {
.param .u32 flag;

)ptx";
    for (int i = 0; i < 1025; ++i) oversized += "st.param.u32 [flag], 0;\ncall.uni noop, (flag);\n";
    oversized += "trap;\n}\n";
    auto imported = cumetal::ir::import_ptx(oversized);
    ok &= expect(imported.ok, "oversized call graph is valid GPU IR");
    if (imported.ok) {
        const auto original = cumetal::ir::print(imported.module);
        std::string error;
        ok &= expect(!metal::expand_trap_call_graphs(&imported.module, &error) &&
                     error.find("bounded CFG size") != std::string::npos,
                     "trap call expansion refuses excessive code growth: " + error);
        ok &= expect(cumetal::ir::print(imported.module) == original,
                     "rejected expansion leaves the original module intact");
    }
    return ok ? 0 : 1;
}
