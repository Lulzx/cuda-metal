#include "cumetal/passes/phase1_pipeline.h"

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

bool has_metadata_field(const cumetal::passes::KernelMetadata& metadata, const std::string& key) {
    for (const auto& field : metadata.fields) {
        if (field.key == key) {
            return true;
        }
    }
    return false;
}

bool has_addrspace(const std::vector<cumetal::passes::AddrspaceInstruction>& instructions,
                   const std::string& opcode,
                   int address_space) {
    for (const auto& instruction : instructions) {
        if (instruction.opcode == opcode && instruction.address_space == address_space) {
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

.visible .entry k0(
    .param .u64 p0
)
{
    ret;
}

.visible .entry k1(
    .param .u64 p0
)
{
    mov.u32 %r0, %tid.x;
    call.uni (%r1), vprintf, ("k1=%u", %r0);
    ld.shared.u32 %r1, [%rd1];
    foo.shared.u32 %r2, %r1;
    ret;
}
)PTX";

    cumetal::passes::Phase1PipelineOptions tolerant_options;
    tolerant_options.entry_name = "k1";
    tolerant_options.metadata.air_version = "2.9";
    const auto tolerant = cumetal::passes::run_phase1_pipeline(ptx, tolerant_options);
    if (!expect(tolerant.ok, "tolerant pipeline succeeds")) {
        return 1;
    }
    if (!expect(tolerant.entry_name == "k1", "selected entry is k1")) {
        return 1;
    }
    if (!expect(!tolerant.lowered_instructions.empty(), "lowered instructions emitted")) {
        return 1;
    }
    if (!expect(!tolerant.addrspace_instructions.empty(), "addrspace instructions emitted")) {
        return 1;
    }
    if (!expect(tolerant.printf_calls.size() == 1, "printf lowering emits one call")) {
        return 1;
    }
    if (!expect(tolerant.printf_formats.size() == 1, "printf format table contains one entry")) {
        return 1;
    }
    if (!expect(tolerant.printf_calls[0].format_id == 0, "printf call references format id 0")) {
        return 1;
    }
    if (!expect(has_addrspace(tolerant.addrspace_instructions, "llvm.load", 3),
                "ld.shared mapped to addrspace 3")) {
        return 1;
    }
    if (!expect(tolerant.metadata.kernel_name == "k1", "metadata kernel name")) {
        return 1;
    }
    if (!expect(has_metadata_field(tolerant.metadata, "air.kernel"), "metadata contains air.kernel")) {
        return 1;
    }
    if (!expect(has_metadata_field(tolerant.metadata, "kernel.arg_count"),
                "metadata contains arg count")) {
        return 1;
    }
    if (!expect(has_metadata_field(tolerant.metadata, "kernel.printf.count"),
                "metadata contains printf count")) {
        return 1;
    }
    if (!expect(has_metadata_field(tolerant.metadata, "kernel.printf.0.token"),
                "metadata contains printf token")) {
        return 1;
    }
    if (!expect(!tolerant.warnings.empty(), "tolerant pipeline reports warnings")) {
        return 1;
    }

    cumetal::passes::Phase1PipelineOptions strict_options;
    strict_options.strict = true;
    strict_options.entry_name = "k1";
    const auto strict = cumetal::passes::run_phase1_pipeline(ptx, strict_options);
    if (!expect(!strict.ok, "strict pipeline fails on unsupported opcode")) {
        return 1;
    }
    if (!expect(strict.error.find("failed") != std::string::npos, "strict error has stage context")) {
        return 1;
    }

    const std::string mixed_entries_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry unsupported_neighbor()
{
    prototype_7 : .callprototype ()_ (.param .b64 _);
    ret;
}
.visible .entry selected_kernel()
{
    ret;
}
)PTX";
    cumetal::passes::Phase1PipelineOptions selected_options;
    selected_options.strict = true;
    selected_options.entry_name = "selected_kernel";
    const auto selected =
        cumetal::passes::run_phase1_pipeline(mixed_entries_ptx, selected_options);
    if (!expect(selected.ok, "strict pipeline ignores unsupported unselected entries")) {
        return 1;
    }

    cumetal::passes::Phase1PipelineOptions missing_entry;
    missing_entry.entry_name = "does_not_exist";
    const auto missing = cumetal::passes::run_phase1_pipeline(ptx, missing_entry);
    if (!expect(!missing.ok, "missing entry fails")) {
        return 1;
    }
    if (!expect(missing.error.find("entry not found") != std::string::npos, "missing entry error")) {
        return 1;
    }

    std::printf("PASS: phase1 pipeline unit tests\n");
    return 0;
}
