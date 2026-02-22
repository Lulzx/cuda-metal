#include "cumetal/passes/intrinsic_lower.h"

#include <utility>

namespace cumetal::passes {
namespace {

bool map_special_register_mov(const cumetal::ptx::EntryFunction::Instruction& instruction,
                              LoweredInstruction* lowered) {
    if (lowered == nullptr || instruction.opcode.rfind("mov", 0) != 0 || instruction.operands.size() < 2) {
        return false;
    }

    const std::string& src = instruction.operands[1];
    std::string mapped;
    if (src == "%tid.x") {
        mapped = "air.thread_position_in_threadgroup.x";
    } else if (src == "%tid.y") {
        mapped = "air.thread_position_in_threadgroup.y";
    } else if (src == "%tid.z") {
        mapped = "air.thread_position_in_threadgroup.z";
    } else if (src == "%ctaid.x") {
        mapped = "air.threadgroup_position_in_grid.x";
    } else if (src == "%ctaid.y") {
        mapped = "air.threadgroup_position_in_grid.y";
    } else if (src == "%ctaid.z") {
        mapped = "air.threadgroup_position_in_grid.z";
    } else if (src == "%ntid.x") {
        mapped = "air.threads_per_threadgroup.x";
    } else if (src == "%ntid.y") {
        mapped = "air.threads_per_threadgroup.y";
    } else if (src == "%ntid.z") {
        mapped = "air.threads_per_threadgroup.z";
    } else if (src == "%nctaid.x") {
        mapped = "air.threadgroups_per_grid.x";
    } else if (src == "%nctaid.y") {
        mapped = "air.threadgroups_per_grid.y";
    } else if (src == "%nctaid.z") {
        mapped = "air.threadgroups_per_grid.z";
    } else if (src == "%laneid") {
        // Lane index within SIMD-group (warp). Equivalent to thread_index_in_simdgroup.
        mapped = "air.thread_position_in_simdgroup";
    } else if (src == "%warpsize") {
        // Architecturally fixed at 32 on all Apple Silicon (spec §7).
        mapped = "air.constant.warp_size";
    } else if (src == "%lanemask_eq") {
        // Bitmask with only the current lane's bit set: (1u << laneid).
        mapped = "air.simdgroup.lanemask_eq";
    } else if (src == "%lanemask_lt") {
        // Bits set for all lanes with index < laneid: (1u << laneid) - 1.
        mapped = "air.simdgroup.lanemask_lt";
    } else if (src == "%lanemask_le") {
        // Bits set for all lanes with index <= laneid.
        mapped = "air.simdgroup.lanemask_le";
    } else if (src == "%lanemask_gt") {
        // Bits set for all lanes with index > laneid.
        mapped = "air.simdgroup.lanemask_gt";
    } else if (src == "%lanemask_ge") {
        // Bits set for all lanes with index >= laneid.
        mapped = "air.simdgroup.lanemask_ge";
    } else {
        return false;
    }

    lowered->opcode = "air.read.special_register";
    lowered->operands = {instruction.operands[0], mapped};
    lowered->translated = true;
    return true;
}

bool map_barrier(const cumetal::ptx::EntryFunction::Instruction& instruction, LoweredInstruction* lowered) {
    if (lowered == nullptr) {
        return false;
    }
    const std::string& op = instruction.opcode;

    if (op.rfind("bar.sync", 0) == 0) {
        lowered->opcode = "air.threadgroup_barrier";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // bar.arrive N, count — arrive at barrier without waiting (cooperative groups)
    // Conservative lowering: full threadgroup barrier (safe superset of arrive semantics).
    if (op.rfind("bar.arrive", 0) == 0) {
        lowered->opcode = "air.threadgroup_barrier";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // bar.red: barrier-level predicate reductions (sync + reduce at named barrier)
    // bar.red.and.pred d, barrier, count, pred → sync and check all pred
    // bar.red.or.pred  d, barrier, count, pred → sync and check any pred
    // bar.red.popc.u32 d, barrier, count, pred → sync and count threads with pred
    // Conservative: map to simdgroup-level operations (threadgroup barrier + reduce).
    if (op.rfind("bar.red", 0) == 0) {
        if (op.find(".and.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.all";
        } else if (op.find(".or.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.any";
        } else if (op.find(".popc.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.reduce_add";
        } else {
            lowered->opcode = "air.threadgroup_barrier";
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // prefetch / prefetchu — cache prefetch hints (memory latency optimization)
    // On Apple Silicon UMA, the memory subsystem handles caching automatically.
    // Lower to no-op (empty barrier-free hint marker).
    if (op.rfind("prefetch", 0) == 0) {
        lowered->opcode = "air.prefetch.noop";
        lowered->operands = {};
        lowered->translated = true;
        return true;
    }

    // membar.gl  → device-wide fence  (__threadfence)
    // membar.sys → system-wide fence  (__threadfence_system)
    // membar.cta → threadgroup fence  (__threadfence_block)
    if (op.rfind("membar", 0) == 0) {
        lowered->opcode = (op.find(".cta") != std::string::npos)
                              ? "air.mem.barrier.threadgroup"
                              : "air.mem.barrier.device";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // fence: Ampere+ memory fence (ISA 7.0+)
    // fence.sc.{cta,gpu,sys} → threadgroup or device barrier
    if (op.rfind("fence", 0) == 0) {
        lowered->opcode = (op.find(".cta") != std::string::npos)
                              ? "air.mem.barrier.threadgroup"
                              : "air.mem.barrier.device";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // red: global/shared reduction (write-only atomic; no return value)
    // red.global.add.f32 [addr], src → atomic add with result discarded
    // Lowered to air.atomic.reduce (same as atomicrmw but no destination).
    // Note: must check "red.global"/"red.shared" (not just "red") to avoid
    // matching "redux.*" which starts with the same prefix.
    if (op.rfind("red.global", 0) == 0 || op.rfind("red.shared", 0) == 0) {
        if (op.find(".shared") != std::string::npos) {
            lowered->opcode = "llvm.atomicrmw.noret.shared";
        } else {
            lowered->opcode = "llvm.atomicrmw.noret.global";
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    return false;
}

bool map_async_copy(const cumetal::ptx::EntryFunction::Instruction& instruction,
                    LoweredInstruction* lowered) {
    if (lowered == nullptr || instruction.opcode.rfind("cp.async", 0) != 0) {
        return false;
    }
    const std::string& op = instruction.opcode;

    // cp.async.commit_group / cp.async.wait_group / cp.async.wait_all →
    // threadgroup barrier (serializes the async copy pipeline;
    // functional but not performance-equivalent to hardware async copy)
    if (op.find("commit_group") != std::string::npos ||
        op.find("wait_group") != std::string::npos ||
        op.find("wait_all") != std::string::npos) {
        lowered->opcode = "air.threadgroup_barrier";
        lowered->operands = {};
        lowered->translated = true;
        return true;
    }

    // cp.async.ca.shared.global [dst], [src], size →
    // marked as air.cp_async; downstream stages lower to synchronous ld+st
    lowered->opcode = "air.cp_async";
    lowered->operands = instruction.operands;
    lowered->translated = true;
    return true;
}

bool map_math(const cumetal::ptx::EntryFunction::Instruction& instruction, LoweredInstruction* lowered) {
    if (lowered == nullptr) {
        return false;
    }

    if (instruction.opcode.rfind("add", 0) == 0) {
        lowered->opcode = "llvm.add";
    } else if (instruction.opcode.rfind("sub", 0) == 0) {
        lowered->opcode = "llvm.sub";
    } else if (instruction.opcode.rfind("mul", 0) == 0) {
        lowered->opcode = "llvm.mul";
    } else if (instruction.opcode.rfind("div", 0) == 0) {
        lowered->opcode = "llvm.div";
    } else if (instruction.opcode.rfind("rem", 0) == 0) {
        lowered->opcode = "llvm.rem";
    } else if (instruction.opcode.rfind("and", 0) == 0) {
        lowered->opcode = "llvm.and";
    } else if (instruction.opcode.rfind("or", 0) == 0) {
        lowered->opcode = "llvm.or";
    } else if (instruction.opcode.rfind("xor", 0) == 0) {
        lowered->opcode = "llvm.xor";
    } else if (instruction.opcode.rfind("not", 0) == 0) {
        lowered->opcode = "llvm.not";
    } else if (instruction.opcode.rfind("shl", 0) == 0) {
        lowered->opcode = "llvm.shl";
    } else if (instruction.opcode.rfind("shr", 0) == 0) {
        lowered->opcode = "llvm.shr";
    } else if (instruction.opcode.rfind("selp", 0) == 0) {
        lowered->opcode = "llvm.select";
    } else if (instruction.opcode.rfind("rcp", 0) == 0) {
        lowered->opcode = "llvm.rcp";
    } else if (instruction.opcode.rfind("neg", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fneg" : "llvm.neg";
    } else if (instruction.opcode.rfind("fma", 0) == 0) {
        lowered->opcode = "llvm.fma";
    } else if (instruction.opcode.rfind("mad", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fma" : "llvm.mad";
    } else if (instruction.opcode.rfind("max", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fmax" : "llvm.max";
    } else if (instruction.opcode.rfind("min", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fmin" : "llvm.min";
    } else if (instruction.opcode.rfind("abs", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fabs" : "llvm.abs";
    } else if (instruction.opcode.rfind("sqrt", 0) == 0) {
        lowered->opcode = "llvm.sqrt";
    } else if (instruction.opcode.rfind("rsqrt", 0) == 0) {
        lowered->opcode = "llvm.rsqrt";
    } else if (instruction.opcode.rfind("ex2", 0) == 0) {
        lowered->opcode = "llvm.exp2";
    } else if (instruction.opcode.rfind("lg2", 0) == 0) {
        lowered->opcode = "llvm.log2";
    } else if (instruction.opcode.rfind("sin", 0) == 0) {
        lowered->opcode = "llvm.sin";
    } else if (instruction.opcode.rfind("cos", 0) == 0) {
        lowered->opcode = "llvm.cos";
    } else if (instruction.opcode.rfind("clz", 0) == 0) {
        // clz.b32 dst, src → @llvm.ctlz.i32(i32, i1 false)
        // clz.b64 dst, src → @llvm.ctlz.i64(i64, i1 false)
        const bool is_64 = instruction.opcode.find(".b64") != std::string::npos;
        lowered->opcode = is_64 ? "llvm.ctlz.i64" : "llvm.ctlz.i32";
    } else if (instruction.opcode.rfind("popc", 0) == 0) {
        // popc.b32 dst, src → @llvm.ctpop.i32(i32)
        // popc.b64 dst, src → @llvm.ctpop.i64(i64)
        const bool is_64 = instruction.opcode.find(".b64") != std::string::npos;
        lowered->opcode = is_64 ? "llvm.ctpop.i64" : "llvm.ctpop.i32";
    } else if (instruction.opcode.rfind("brev", 0) == 0) {
        // brev.b32 dst, src → @llvm.bitreverse.i32(i32)
        // brev.b64 dst, src → @llvm.bitreverse.i64(i64)
        const bool is_64 = instruction.opcode.find(".b64") != std::string::npos;
        lowered->opcode = is_64 ? "llvm.bitreverse.i64" : "llvm.bitreverse.i32";
    } else if (instruction.opcode.rfind("isspacep", 0) == 0) {
        // isspacep.{global,shared,local,const} pred, ptr
        // On Apple Silicon, all device memory is in the global address space (flat UMA).
        // Conservative lowering: isspacep.global → true; all others → false.
        const bool is_global = instruction.opcode.find(".global") != std::string::npos;
        lowered->opcode = is_global ? "air.isspacep.global" : "air.isspacep.nonglobal";
    } else if (instruction.opcode.rfind("bfe", 0) == 0) {
        // bfe.{u32,s32,u64,s64} d, a, b, c  — bit field extract
        // Lowered to air.bfe; downstream stages emit shift+mask sequences.
        const bool is_signed = instruction.opcode.find(".s") != std::string::npos;
        lowered->opcode = is_signed ? "air.bfe.signed" : "air.bfe.unsigned";
    } else if (instruction.opcode.rfind("bfi", 0) == 0) {
        // bfi.b32 d, a, b, c, f  — bit field insert
        lowered->opcode = "air.bfi";
    } else if (instruction.opcode.rfind("prmt", 0) == 0) {
        // prmt.b32 d, a, b, c  — byte permutation from two 32-bit sources
        // Lowered to air.prmt; downstream stages emit byte-select sequences.
        lowered->opcode = "air.prmt";
    } else if (instruction.opcode.rfind("match", 0) == 0) {
        // match.any.sync.b32 d, a, mask — for each bit in mask: 1 if that lane has same a
        // match.all.sync.b32 d, p, a, mask — 1 if all masked lanes have same a
        // Conservative lowering: full-group match via air.simdgroup comparison.
        if (instruction.opcode.find(".any.") != std::string::npos) {
            lowered->opcode = "air.match.any.sync";
        } else if (instruction.opcode.find(".all.") != std::string::npos) {
            lowered->opcode = "air.match.all.sync";
        } else {
            lowered->opcode = "air.match.sync";
        }
    } else if (instruction.opcode.rfind("nanosleep", 0) == 0) {
        // nanosleep.u32 delay — busy-wait for ~delay nanoseconds (Pascal+)
        // On Metal, lowered to a conservative no-op (Metal has no sleep primitive).
        lowered->opcode = "air.nanosleep";
    } else if (instruction.opcode.rfind("trap", 0) == 0) {
        // trap — raise a hardware trap (abort the thread)
        // Lowered to @llvm.trap() which emits an illegal instruction.
        lowered->opcode = "llvm.trap";
    } else if (instruction.opcode.rfind("exit", 0) == 0) {
        // exit — terminate the current thread (same as ret in our model)
        lowered->opcode = "air.exit";
    } else if (instruction.opcode.rfind("activemask", 0) == 0) {
        // activemask.b32 d  — bitmask of all currently active threads in the warp
        // On Apple Silicon, SIMD-groups are always fully active (no divergence in our model).
        // Lowered to air.activemask → constant 0xFFFFFFFF (all 32 lanes active).
        lowered->opcode = "air.activemask";
    } else if (instruction.opcode.rfind("fns", 0) == 0) {
        // fns.b32 d, membermask, base, offset — find the Nth set bit in membermask
        // counting from position 'base', stepping by 'offset'.
        // Used in warp-convergence algorithms. Lowered to air.fns; downstream
        // stages implement via popcount + partial mask operations.
        lowered->opcode = "air.fns";
    } else if (instruction.opcode.rfind("bfind", 0) == 0) {
        // bfind.{u32,s32,u64} d, a  — find most significant non-sign bit
        // bfind.shiftamt.* returns 0 when d==0; plain bfind returns 0xffffffff.
        // Equivalent to: 31 - clz(a) (for u32/s32); 63 - clz(a) (for u64).
        // Lowered to air.bfind; downstream stages emit ctlz + sub.
        const bool is_64 = instruction.opcode.find(".u64") != std::string::npos ||
                           instruction.opcode.find(".s64") != std::string::npos;
        lowered->opcode = is_64 ? "air.bfind.i64" : "air.bfind.i32";
    } else if (instruction.opcode.rfind("lop3", 0) == 0) {
        // lop3.b32 d, a, b, c, immLut  — 3-input look-up-table logic (Turing+)
        // Maps to air.lop3; downstream stages lower to a sequence of and/or/xor/not
        // operations implementing the LUT entry.
        lowered->opcode = "air.lop3";
    } else if (instruction.opcode.rfind("sad", 0) == 0) {
        // sad.{u32,s32} d, a, b, c  — sum of absolute differences: d = |a-b| + c
        // Lowered to air.sad; downstream stages emit abs(sub(a,b))+c sequence.
        lowered->opcode = "air.sad";
    } else if (instruction.opcode.rfind("testp", 0) == 0) {
        // testp.{finite,infinite,number,notanumber,normal,subnormal,notfinite}.{f32,f64} pred, src
        // Tests floating-point properties (equivalent to C isfinite/isinf/isnan/etc.).
        // Lowered to air.testp.*; downstream stages emit llvm.is_fpclass calls.
        const std::string& op = instruction.opcode;
        if (op.find(".finite") != std::string::npos) {
            lowered->opcode = "air.testp.finite";
        } else if (op.find(".infinite") != std::string::npos) {
            lowered->opcode = "air.testp.infinite";
        } else if (op.find(".notanumber") != std::string::npos ||
                   op.find(".snan") != std::string::npos) {
            lowered->opcode = "air.testp.nan";
        } else if (op.find(".number") != std::string::npos) {
            lowered->opcode = "air.testp.number";
        } else if (op.find(".subnormal") != std::string::npos) {
            lowered->opcode = "air.testp.subnormal";
        } else if (op.find(".normal") != std::string::npos) {
            lowered->opcode = "air.testp.normal";
        } else if (op.find(".notfinite") != std::string::npos) {
            lowered->opcode = "air.testp.notfinite";
        } else {
            lowered->opcode = "air.testp";
        }
    } else {
        return false;
    }

    lowered->operands = instruction.operands;
    lowered->translated = true;
    return true;
}

bool map_warp_primitives(const cumetal::ptx::EntryFunction::Instruction& instruction,
                          LoweredInstruction* lowered) {
    if (lowered == nullptr) {
        return false;
    }

    const std::string& op = instruction.opcode;

    // shfl: shuffle within a SIMD-group (warp)
    // shfl.sync.{idx,down,up,bfly}.b32 dst, src, sel, clamp, membermask
    if (op.rfind("shfl", 0) == 0) {
        if (op.find(".idx.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle";
        } else if (op.find(".down.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_down";
        } else if (op.find(".up.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_up";
        } else if (op.find(".bfly.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_xor";
        } else {
            return false;
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // vote: warp-wide predicate reduction (ballot / any / all / uni)
    // vote.[sync.]{ballot.b32,any.pred,all.pred,uni.pred}
    if (op.rfind("vote", 0) == 0) {
        if (op.find(".ballot.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.ballot";
        } else if (op.find(".any.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.any";
        } else if (op.find(".all.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.all";
        } else if (op.find(".uni.") != std::string::npos) {
            // vote.uni.pred d, pred — 1 if all active threads have same pred value
            // Lowered to air.simdgroup.all (conservative: true only if all agree AND
            // all are active). Matches CUDA semantics for a uniform predicate.
            lowered->opcode = "air.simdgroup.all";
        } else {
            return false;
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // bar.warp.sync → simdgroup barrier (__syncwarp)
    if (op.rfind("bar.warp.sync", 0) == 0) {
        lowered->opcode = "air.simdgroup.barrier";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // redux.sync: warp-wide reduction (__redux_sync, Ampere+)
    // redux.sync.{add,and,or,xor,min,max}.{s32,u32,b32,f32} dst, src, membermask
    if (op.rfind("redux.sync", 0) == 0) {
        const bool is_float = op.find(".f32") != std::string::npos;
        if (op.find(".add.") != std::string::npos) {
            lowered->opcode = is_float ? "air.simdgroup.reduce_add.f32" : "air.simdgroup.reduce_add";
        } else if (op.find(".and.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.reduce_and";
        } else if (op.find(".or.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.reduce_or";
        } else if (op.find(".xor.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.reduce_xor";
        } else if (op.find(".min.") != std::string::npos) {
            lowered->opcode = is_float ? "air.simdgroup.reduce_min.f32" : "air.simdgroup.reduce_min";
        } else if (op.find(".max.") != std::string::npos) {
            lowered->opcode = is_float ? "air.simdgroup.reduce_max.f32" : "air.simdgroup.reduce_max";
        } else {
            return false;
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    return false;
}

}  // namespace

IntrinsicLowerResult lower_intrinsics(const cumetal::ptx::EntryFunction& entry,
                                      const IntrinsicLowerOptions& options) {
    IntrinsicLowerResult result;
    for (const auto& instruction : entry.instructions) {
        LoweredInstruction lowered;
        lowered.opcode = instruction.opcode;
        lowered.operands = instruction.operands;
        lowered.translated = false;

        if (instruction.opcode == "ptx.label") {
            lowered.translated = true;
            result.instructions.push_back(std::move(lowered));
            continue;
        }

        bool translated = false;
        translated = translated || map_special_register_mov(instruction, &lowered);
        translated = translated || map_barrier(instruction, &lowered);
        translated = translated || map_async_copy(instruction, &lowered);
        translated = translated || map_warp_primitives(instruction, &lowered);
        translated = translated || map_math(instruction, &lowered);

        if (!translated) {
            if (instruction.opcode.rfind("ret", 0) == 0 || instruction.opcode.rfind("ld", 0) == 0 ||
                instruction.opcode.rfind("st", 0) == 0 || instruction.opcode.rfind("setp", 0) == 0 ||
                instruction.opcode.rfind("bra", 0) == 0 || instruction.opcode.rfind("cvt", 0) == 0 ||
                instruction.opcode.rfind("cvta", 0) == 0 || instruction.opcode.rfind("mov", 0) == 0 ||
                instruction.opcode.rfind("call", 0) == 0 || instruction.opcode.rfind("atom", 0) == 0 ||
                instruction.opcode.rfind("selp", 0) == 0 || instruction.opcode.rfind("set.", 0) == 0) {
                // keep instruction as-is; lowering not needed yet for this stage.
            } else {
                result.warnings.push_back("intrinsic_lower: no mapping for opcode '" + instruction.opcode + "'");
            }
        }

        result.instructions.push_back(std::move(lowered));
    }

    if (options.strict && !result.warnings.empty()) {
        result.error = result.warnings.front();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
