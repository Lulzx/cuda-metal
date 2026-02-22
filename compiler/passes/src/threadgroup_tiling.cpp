#include "cumetal/passes/threadgroup_tiling.h"

#include <cstdlib>
#include <set>
#include <sstream>
#include <utility>

namespace cumetal::passes {
namespace {

// Apple Silicon Metal threadgroup memory has 32 banks, 4 bytes per bank.
constexpr std::uint32_t kBankCount      = 32;
constexpr std::uint32_t kBankWidthBytes = 4;
// kBankWidthBits = kBankCount * kBankWidthBytes * 8 = 1024 (kept for reference)

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

// Extract the element type from a PTX opcode suffix such as "ld.shared.f32".
SharedMemElemType elem_type_from_opcode(const std::string& opcode) {
    // Check suffixes from longest to shortest to avoid false matches.
    if (opcode.find(".b128") != std::string::npos ||
        opcode.find(".v4.f32") != std::string::npos ||
        opcode.find(".v4.u32") != std::string::npos) {
        return SharedMemElemType::kB128;
    }
    if (opcode.find(".f64") != std::string::npos ||
        opcode.find(".u64") != std::string::npos ||
        opcode.find(".s64") != std::string::npos ||
        opcode.find(".b64") != std::string::npos) {
        return SharedMemElemType::kB64;
    }
    if (opcode.find(".f32") != std::string::npos ||
        opcode.find(".u32") != std::string::npos ||
        opcode.find(".s32") != std::string::npos ||
        opcode.find(".b32") != std::string::npos) {
        return SharedMemElemType::kB32;
    }
    if (opcode.find(".f16") != std::string::npos ||
        opcode.find(".u16") != std::string::npos ||
        opcode.find(".s16") != std::string::npos ||
        opcode.find(".b16") != std::string::npos) {
        return SharedMemElemType::kB16;
    }
    if (opcode.find(".u8") != std::string::npos ||
        opcode.find(".s8") != std::string::npos ||
        opcode.find(".b8") != std::string::npos) {
        return SharedMemElemType::kB8;
    }
    return SharedMemElemType::kUnknown;
}

std::size_t elem_bytes_for(SharedMemElemType t) {
    switch (t) {
        case SharedMemElemType::kB8:   return 1;
        case SharedMemElemType::kB16:  return 2;
        case SharedMemElemType::kB32:  return 4;
        case SharedMemElemType::kB64:  return 8;
        case SharedMemElemType::kB128: return 16;
        case SharedMemElemType::kUnknown: return 0;
    }
    return 0;
}

bool is_shared_mem_op(const std::string& opcode) {
    return starts_with(opcode, "ld.shared") ||
           starts_with(opcode, "st.shared") ||
           starts_with(opcode, "atom.shared") ||
           starts_with(opcode, "red.shared");
}

// Return true if opcode is a mul.lo or shl that could produce a stride.
// Extract the constant operand value into *out_value.
bool extract_stride_from_instruction(const cumetal::ptx::EntryFunction::Instruction& inst,
                                     std::uint32_t* out_value) {
    // mul.lo.u32 %dst, %src, <imm>  — last operand is the multiplier
    if (starts_with(inst.opcode, "mul.lo") ||
        starts_with(inst.opcode, "mul.wide") ||
        starts_with(inst.opcode, "mul.hi")) {
        // operands: [dst, src1, src2]  — we look for a numeric immediate in src2 or src1
        for (std::size_t i = inst.operands.size(); i-- > 0; ) {
            const auto& op = inst.operands[i];
            // Skip register operands (start with % or contain letters).
            bool all_digits = !op.empty();
            for (char c : op) {
                if (c < '0' || c > '9') { all_digits = false; break; }
            }
            if (all_digits && !op.empty()) {
                const auto v = static_cast<std::uint32_t>(std::strtoul(op.c_str(), nullptr, 10));
                if (v > 0) {
                    *out_value = v;
                    return true;
                }
            }
        }
        return false;
    }

    // shl.b32 %dst, %src, <shift>  — last operand is shift amount
    if (starts_with(inst.opcode, "shl.b32") ||
        starts_with(inst.opcode, "shl.b64") ||
        starts_with(inst.opcode, "shl.u32") ||
        starts_with(inst.opcode, "shl.u64")) {
        if (!inst.operands.empty()) {
            const auto& last = inst.operands.back();
            bool all_digits = !last.empty();
            for (char c : last) {
                if (c < '0' || c > '9') { all_digits = false; break; }
            }
            if (all_digits && !last.empty()) {
                const auto shift = static_cast<std::uint32_t>(std::strtoul(last.c_str(), nullptr, 10));
                if (shift > 0 && shift < 32) {
                    *out_value = 1u << shift;  // shl by N  ≡  multiply by 2^N
                    return true;
                }
            }
        }
        return false;
    }

    return false;
}

// Returns true if stride is a power of 2 ≥ 16 (i.e., at least 16 elements).
bool is_conflict_prone_stride(std::uint32_t stride) {
    return stride >= 16 && (stride & (stride - 1)) == 0;
}

// Given a power-of-2 stride and element size, determine if a bank conflict
// occurs in a 32-bank, 4-byte-per-bank system.
//
// A bank conflict occurs when N threads access elements separated by a stride
// that is a multiple of 32 banks × bank_width_bytes / elem_bytes banks.
// For 4-byte elements: conflict when stride is a multiple of 32.
// For 2-byte elements: conflict when stride is a multiple of 64.
// For 8-byte elements: conflict when stride is a multiple of 16.
// For 1-byte elements: conflict when stride is a multiple of 128.
bool has_bank_conflict(std::uint32_t stride_elements, std::size_t elem_bytes) {
    if (elem_bytes == 0) return false;
    const std::uint32_t bank_period =
        (kBankCount * kBankWidthBytes) / static_cast<std::uint32_t>(elem_bytes);
    return (stride_elements % bank_period) == 0;
}

// Compute the recommended padding (in bytes) to break the bank conflict.
// Adding 1 element of padding makes the stride non-divisible by bank_period.
std::uint32_t recommended_padding_bytes(std::size_t elem_bytes) {
    // Padding one element always breaks the alignment, regardless of element size.
    return static_cast<std::uint32_t>(elem_bytes);
}

}  // namespace

TilingAnalysis analyse_threadgroup_tiling(const cumetal::ptx::EntryFunction& entry) {
    TilingAnalysis result;

    // Rolling window of stride hints from the last few instructions.
    // We keep up to 4 preceding mul/shl results as potential stride sources.
    constexpr int kWindowSize = 4;
    struct RecentStride { std::uint32_t value; int line; };
    std::vector<RecentStride> recent_strides;
    recent_strides.reserve(kWindowSize);

    // Track which (stride_elements, elem_bytes) pairs we have already emitted
    // a hint for, to avoid duplicate hints.
    std::set<std::pair<std::uint32_t, std::size_t>> emitted;

    for (const auto& inst : entry.instructions) {
        // Try to harvest a stride constant from mul/shl instructions.
        std::uint32_t stride_val = 0;
        if (extract_stride_from_instruction(inst, &stride_val)) {
            if (static_cast<int>(recent_strides.size()) >= kWindowSize) {
                recent_strides.erase(recent_strides.begin());
            }
            recent_strides.push_back({stride_val, inst.line});

            StrideHint sh;
            sh.stride_value = stride_val;
            sh.line = inst.line;
            result.stride_hints.push_back(sh);
        }

        if (!is_shared_mem_op(inst.opcode)) {
            continue;
        }

        result.uses_shared_memory = true;

        SharedMemAccess access;
        access.opcode     = inst.opcode;
        access.elem_type  = elem_type_from_opcode(inst.opcode);
        access.elem_bytes = elem_bytes_for(access.elem_type);
        access.is_write   = starts_with(inst.opcode, "st.shared") ||
                            starts_with(inst.opcode, "atom.shared") ||
                            starts_with(inst.opcode, "red.shared");
        access.line = inst.line;
        result.accesses.push_back(access);

        if (access.elem_bytes == 0) {
            result.warnings.push_back("threadgroup_tiling: unknown element type in '" +
                                      inst.opcode + "' at line " + std::to_string(inst.line));
            continue;
        }

        // Check recent strides against this access for bank conflict risk.
        for (const auto& rs : recent_strides) {
            if (!is_conflict_prone_stride(rs.value)) {
                continue;
            }
            if (!has_bank_conflict(rs.value, access.elem_bytes)) {
                continue;
            }

            const auto key = std::make_pair(rs.value, access.elem_bytes);
            if (emitted.count(key)) {
                continue;
            }
            emitted.insert(key);

            TilingHint hint;
            hint.detected_stride = rs.value;
            hint.elem_bytes      = access.elem_bytes;
            hint.padding_bytes   = recommended_padding_bytes(access.elem_bytes);
            hint.needs_padding   = true;

            std::ostringstream oss;
            oss << "stride " << rs.value << " elements × "
                << access.elem_bytes << "B = "
                << (rs.value * access.elem_bytes) << "B aligns to "
                << kBankCount << "-bank boundary; add "
                << hint.padding_bytes << "B padding per row";
            hint.reason = oss.str();

            result.hints.push_back(hint);
        }
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
