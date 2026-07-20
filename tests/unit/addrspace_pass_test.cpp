#include "cumetal/passes/addrspace.h"
#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
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
    ld.shared.u32 %r0, [%rd1];
    st.shared.u32 [%rd1], %r0;
    ld.global.f32 %f0, [%rd2];
    st.global.f32 [%rd2], %f0;
    ld.local.u16 %r1, [%rd3];
    st.local.u16 [%rd3], %r1;
    cvta.to.shared.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.local.u64 %rd6, %rd3;
    ld.volatile.shared.u32 %r9, [%rd1];
    st.volatile.shared.u32 [%rd1], %r9;
    foo.shared.u32 %r7, %r8;
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(ptx);
    if (!expect(parsed.ok, "parse PTX for addrspace pass")) {
        return 1;
    }

    const auto rewritten = cumetal::passes::rewrite_addrspace(parsed.module.entries[0]);
    if (!expect(rewritten.ok, "non-strict addrspace rewrite succeeds")) {
        return 1;
    }
    if (!expect(rewritten.instructions.size() == parsed.module.entries[0].instructions.size(),
                "instruction count preserved")) {
        return 1;
    }

    if (!expect(rewritten.instructions[0].rewritten &&
                    rewritten.instructions[0].opcode == "llvm.load" &&
                    rewritten.instructions[0].address_space == 3,
                "ld.shared rewritten to as3 load")) {
        return 1;
    }
    if (!expect(rewritten.instructions[1].rewritten &&
                    rewritten.instructions[1].opcode == "llvm.store" &&
                    rewritten.instructions[1].address_space == 3,
                "st.shared rewritten to as3 store")) {
        return 1;
    }
    if (!expect(rewritten.instructions[2].rewritten &&
                    rewritten.instructions[2].address_space == 1,
                "ld.global rewritten to as1")) {
        return 1;
    }
    if (!expect(rewritten.instructions[3].rewritten &&
                    rewritten.instructions[3].address_space == 1,
                "st.global rewritten to as1")) {
        return 1;
    }
    if (!expect(rewritten.instructions[4].rewritten &&
                    rewritten.instructions[4].address_space == 5,
                "ld.local rewritten to as5")) {
        return 1;
    }
    if (!expect(rewritten.instructions[5].rewritten &&
                    rewritten.instructions[5].address_space == 5,
                "st.local rewritten to as5")) {
        return 1;
    }
    if (!expect(rewritten.instructions[6].rewritten &&
                    rewritten.instructions[6].opcode == "llvm.addrspacecast.to.as3",
                "cvta shared rewritten")) {
        return 1;
    }
    if (!expect(rewritten.instructions[7].rewritten &&
                    rewritten.instructions[7].opcode == "llvm.addrspacecast.to.as1",
                "cvta global rewritten")) {
        return 1;
    }
    if (!expect(rewritten.instructions[8].rewritten &&
                    rewritten.instructions[8].opcode == "llvm.addrspacecast.to.as5",
                "cvta local rewritten")) {
        return 1;
    }
    if (!expect(rewritten.instructions[9].rewritten &&
                    rewritten.instructions[9].opcode == "llvm.load" &&
                    rewritten.instructions[9].address_space == 3,
                "qualified volatile shared load rewritten")) {
        return 1;
    }
    if (!expect(rewritten.instructions[10].rewritten &&
                    rewritten.instructions[10].opcode == "llvm.store" &&
                    rewritten.instructions[10].address_space == 3,
                "qualified volatile shared store rewritten")) {
        return 1;
    }
    if (!expect(!rewritten.warnings.empty(), "unknown .shared opcode emits warning")) {
        return 1;
    }

    cumetal::passes::AddrspaceRewriteOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::passes::rewrite_addrspace(parsed.module.entries[0], strict_options);
    if (!expect(!strict.ok, "strict addrspace rewrite fails on unknown opcode")) {
        return 1;
    }
    if (!expect(strict.error.find("no rewrite rule") != std::string::npos,
                "strict error reports missing rewrite rule")) {
        return 1;
    }

    std::printf("PASS: addrspace pass unit tests\n");
    return 0;
}
