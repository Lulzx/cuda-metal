#include "cumetal/passes/addrspace.h"

#include <utility>

namespace cumetal::passes {
namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool has_address_space_token(const std::string& opcode) {
    return opcode.find(".shared") != std::string::npos ||
           opcode.find(".global") != std::string::npos ||
           opcode.find(".local") != std::string::npos ||
           opcode.find(".const") != std::string::npos;
}

bool rewrite_load_store(const cumetal::ptx::EntryFunction::Instruction& instruction,
                        AddrspaceInstruction* out) {
    if (out == nullptr) {
        return false;
    }

    const bool is_load = starts_with(instruction.opcode, "ld.");
    const bool is_store = starts_with(instruction.opcode, "st.");
    const bool is_atomic = starts_with(instruction.opcode, "atom.");

    if (is_load && instruction.opcode.find(".shared") != std::string::npos) {
        out->opcode = "llvm.load";
        out->address_space = 3;
    } else if (is_load && instruction.opcode.find(".global") != std::string::npos) {
        out->opcode = "llvm.load";
        out->address_space = 1;
    } else if (is_load && instruction.opcode.find(".local") != std::string::npos) {
        out->opcode = "llvm.load";
        out->address_space = 5;
    } else if (is_store && instruction.opcode.find(".shared") != std::string::npos) {
        out->opcode = "llvm.store";
        out->address_space = 3;
    } else if (is_store && instruction.opcode.find(".global") != std::string::npos) {
        out->opcode = "llvm.store";
        out->address_space = 1;
    } else if (is_store && instruction.opcode.find(".local") != std::string::npos) {
        out->opcode = "llvm.store";
        out->address_space = 5;
    } else if (is_atomic && instruction.opcode.find(".shared") != std::string::npos) {
        out->opcode = "llvm.atomicrmw";
        out->address_space = 3;
    } else if (is_atomic && instruction.opcode.find(".global") != std::string::npos) {
        out->opcode = "llvm.atomicrmw";
        out->address_space = 1;
    } else if (is_load && instruction.opcode.find(".const") != std::string::npos) {
        // Constant address space (CUDA AS 4 → AIR AS 2).
        // Loads from __constant__ variables; mapped to AIR constant buffer.
        out->opcode = "llvm.load";
        out->address_space = 2;
    } else if (starts_with(instruction.opcode, "cp.async")) {
        // cp.async global→shared: intrinsic_lower handles the actual lowering.
        // Passthrough here to suppress the unhandled-addrspace-opcode warning.
        out->opcode = instruction.opcode;
        out->address_space = 1;  // source is global memory
        out->rewritten = false;
    } else if (starts_with(instruction.opcode, "isspacep")) {
        // isspacep.{global,shared,local,const} pred, ptr
        // intrinsic_lower maps this to air.isspacep.*; passthrough here.
        out->opcode = instruction.opcode;
        out->address_space = 0;
        out->rewritten = false;
    } else if (starts_with(instruction.opcode, "prefetch") ||
               starts_with(instruction.opcode, "prefetchu")) {
        // prefetch / prefetchu: cache hint; no address space rewrite needed (no-op on UMA).
        out->opcode = instruction.opcode;
        out->address_space = 0;
        out->rewritten = false;
    } else if (starts_with(instruction.opcode, "red.global")) {
        // red.global.*: write-only atomic reduction (no return value)
        // intrinsic_lower handles the actual lowering; passthrough here.
        out->opcode = instruction.opcode;
        out->address_space = 1;
        out->rewritten = false;
    } else if (starts_with(instruction.opcode, "red.shared")) {
        out->opcode = instruction.opcode;
        out->address_space = 3;
        out->rewritten = false;
    } else {
        return false;
    }

    out->operands = instruction.operands;
    out->rewritten = true;
    return true;
}

bool rewrite_cvta(const cumetal::ptx::EntryFunction::Instruction& instruction, AddrspaceInstruction* out) {
    if (out == nullptr) {
        return false;
    }

    if (starts_with(instruction.opcode, "cvta.to.shared")) {
        out->opcode = "llvm.addrspacecast.to.as3";
        out->address_space = 3;
    } else if (starts_with(instruction.opcode, "cvta.to.global")) {
        out->opcode = "llvm.addrspacecast.to.as1";
        out->address_space = 1;
    } else if (starts_with(instruction.opcode, "cvta.to.local")) {
        out->opcode = "llvm.addrspacecast.to.as5";
        out->address_space = 5;
    } else if (starts_with(instruction.opcode, "cvta.global") ||
               starts_with(instruction.opcode, "cvta.shared") ||
               starts_with(instruction.opcode, "cvta.local")  ||
               starts_with(instruction.opcode, "cvta.const")) {
        // Reverse-direction cvta: convert FROM a specific address space TO generic/flat (AS0).
        // On Apple Silicon UMA all memory is flat, so this is an identity operation.
        // lower_to_llvm.cpp (emit_cvta_instruction) already emits it as a plain copy.
        out->opcode = instruction.opcode;
        out->address_space = 0;
        out->rewritten = false;
    } else {
        return false;
    }

    out->operands = instruction.operands;
    out->rewritten = true;
    return true;
}

}  // namespace

AddrspaceRewriteResult rewrite_addrspace(const cumetal::ptx::EntryFunction& entry,
                                         const AddrspaceRewriteOptions& options) {
    AddrspaceRewriteResult result;
    for (const auto& instruction : entry.instructions) {
        AddrspaceInstruction rewritten;
        rewritten.opcode = instruction.opcode;
        rewritten.operands = instruction.operands;
        rewritten.address_space = 0;
        rewritten.rewritten = false;

        bool mapped = false;
        mapped = mapped || rewrite_load_store(instruction, &rewritten);
        mapped = mapped || rewrite_cvta(instruction, &rewritten);

        if (!mapped && has_address_space_token(instruction.opcode)) {
            result.warnings.push_back("addrspace: no rewrite rule for opcode '" + instruction.opcode + "'");
        }

        result.instructions.push_back(std::move(rewritten));
    }

    if (options.strict && !result.warnings.empty()) {
        result.error = result.warnings.front();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
