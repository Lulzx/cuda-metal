#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

// FP64 compilation mode (see spec §8.1 and --fp64 CLI flag).
enum class Fp64Mode {
    kNative,   // emit AIR FP64 instructions as-is (default)
    kEmulate,  // Dekker FP32-pair decomposition (~44-bit mantissa); default in runtime
    kWarn,     // same as kNative but emit a per-instruction warning for .f64 ops
};

struct LowerToLlvmOptions {
    bool strict = false;
    std::string entry_name;
    std::string module_id = "cumetal.ptx.module";
    std::string target_triple = "air64_v28-apple-macosx26.0.0";
    Fp64Mode fp64_mode = Fp64Mode::kNative;
};

struct LowerToLlvmResult {
    bool ok = false;
    std::string entry_name;
    std::string llvm_ir;
    std::vector<std::string> warnings;
    std::string error;
};

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx,
                                       const LowerToLlvmOptions& options = {});

// Return the total bytes of static __shared__ memory required by the PTX.
// This is needed to call setThreadgroupMemoryLength at kernel launch time.
std::size_t compute_static_shared_bytes(std::string_view ptx,
                                        std::string_view entry_name = {});

}  // namespace cumetal::ptx
