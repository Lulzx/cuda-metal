#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

// FP64 compilation mode (see spec §8.1 and --fp64 CLI flag).
enum class Fp64Mode {
    kNative,   // emit AIR FP64 instructions as-is (fails at Metal pipeline create on Apple GPU)
    kEmulate,  // Dekker FP32-pair ALU + IEEE binary64 register/memory bits; runtime default
    kWide48,   // full-binary64-range scaled FP32-pair arithmetic via VF64 support AIR
    kIEEE64,   // correctly rounded software binary64 via VF64 support AIR
    kWarn,     // same as kNative but emit a per-instruction warning for .f64 ops
};

// Runtime/driver/registration default: emulate unless CUMETAL_FP64_MODE=native|warn.
inline Fp64Mode fp64_mode_from_env() {
    const char* env = std::getenv("CUMETAL_FP64_MODE");
    if (env != nullptr) {
        const std::string_view mode(env);
        if (mode == "native") return Fp64Mode::kNative;
        if (mode == "emulate" || mode == "fast48") return Fp64Mode::kEmulate;
        if (mode == "wide48") return Fp64Mode::kWide48;
        if (mode == "ieee64") return Fp64Mode::kIEEE64;
        if (mode == "warn") return Fp64Mode::kWarn;
    }
    return Fp64Mode::kEmulate;
}

struct LowerToLlvmOptions {
    bool strict = false;
    std::string entry_name;
    std::string module_id = "cumetal.ptx.module";
    std::string target_triple = "air64_v28-apple-macosx26.0.0";
    // Offline cumetalc PTX tools still default to native; runtime JIT overrides via
    // fp64_mode_from_env().
    Fp64Mode fp64_mode = Fp64Mode::kNative;
};

struct LowerToLlvmResult {
    bool ok = false;
    // The kernel performs a 64-bit atomic, which Metal has no instruction for.
    // It therefore takes a hidden `i8 addrspace(1)*` lock-bank parameter that
    // the runtime must bind at kReservedAtomicLockBankIndex.
    bool uses_atomic_lock_bank = false;
    std::string entry_name;
    std::string llvm_ir;
    // Device printf uses two hidden kernel arguments (ring buffer and capacity).
    // The runtime drains records using this format-id table.
    std::vector<std::string> printf_formats;
    // Device malloc/free use one persistent hidden global-memory heap buffer.
    bool uses_device_heap = false;
    // CUDA dynamic parallelism uses a hidden launch queue drained by the host
    // runtime after each parent dispatch.
    bool uses_device_launch_queue = false;
    // PTX clock/globaltimer reads use a hidden device-wide 32-bit atomic
    // counter. Public Metal exposes no shader cycle counter, so this preserves
    // monotonic/wraparound control-flow semantics without claiming timing
    // accuracy.
    bool uses_device_clock = false;
    std::vector<std::string> warnings;
    std::string error;
};

// A module-scope PTX `.const` declaration without an initializer is storage
// supplied by the CUDA registration ABI rather than an LLVM constant global.
// Return only declarations referenced by the selected entry, in declaration
// order, so compiler and runtime agree on hidden Metal buffer bindings.
struct ExternalConstantSymbol {
    std::string name;
    std::size_t offset_bytes = 0;
    std::size_t size_bytes = 0;
};

// Reserved Metal buffer index of the 64-bit atomic lock bank, and the number of
// 32-bit lock words in it. The lowering emits the parameter under a fixed name
// at this index; the backend spots it by reflecting on the compiled function's
// bindings, so a metallib is self-describing and no separate flag can drift.
inline constexpr std::size_t kAtomicLockBankBindingIndex = 29;
inline constexpr std::size_t kAtomicLockBankSlots = 4096;
inline constexpr std::size_t kDeviceClockBindingIndex = 28;
inline constexpr std::size_t kGridBarrierBindingIndex = 27;
inline constexpr std::size_t kGridYOffsetBindingIndex = 26;

std::vector<ExternalConstantSymbol> find_referenced_external_constant_symbols(
    std::string_view ptx,
    std::string_view entry_name);

std::size_t compute_external_constant_buffer_bytes(std::string_view ptx);

using ExternalGlobalSymbol = ExternalConstantSymbol;

std::vector<ExternalGlobalSymbol> find_referenced_external_global_symbols(
    std::string_view ptx,
    std::string_view entry_name);

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx,
                                       const LowerToLlvmOptions& options = {});

// Return the total bytes of static __shared__ memory required by the PTX.
// This is needed to call setThreadgroupMemoryLength at kernel launch time.
std::size_t compute_static_shared_bytes(std::string_view ptx,
                                        std::string_view entry_name = {});

}  // namespace cumetal::ptx
