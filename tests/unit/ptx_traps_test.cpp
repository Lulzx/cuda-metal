#include "cumetal/ir/ptx_importer.h"
#include "trap_reporting.h"
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

std::size_t count_occurrences(const std::string& source,
                              const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t at = 0; (at = source.find(needle, at)) != std::string::npos;
         at += needle.size()) {
        ++count;
    }
    return count;
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
    ok &= expect(helper_trap.ok &&
                     helper_trap.source.find("helper__cm_trap_guarded") !=
                         std::string::npos &&
                     helper_trap.source.find(
                         "atomic_fetch_or_explicit(cm_trap_status") !=
                         std::string::npos,
                 "helper trap uses the shared cancellation ABI: " +
                     helper_trap.error);
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
    ok &= expect(!rejected_barrier.ok &&
                     rejected_barrier.error.find(
                         "barriers, collectives, or printf") !=
                         std::string::npos,
                 "barriers in guarded call graphs remain rejected: " +
                     rejected_barrier.error);
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
    const auto large_graph = metal::compile_ptx_to_msl(oversized);
    ok &= expect(large_graph.ok && large_graph.source.size() < 1000000 &&
                     count_occurrences(large_graph.source,
                                       "void noop__cm_trap_guarded(") == 2,
                 "repeated trap-capable calls preserve one guarded helper: " +
                     large_graph.error);
    return ok ? 0 : 1;
}
