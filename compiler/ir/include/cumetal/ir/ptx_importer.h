#pragma once

#include "cumetal/ir/ir.h"

#include <string>
#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace cumetal::ir {

struct PtxImportOptions {
    bool strict = true;
    std::string entry_name;
    std::string source_name;
    std::string fp64_mode = "fast48";
    // Optional tighter per-function SSA type-proof work limit. Zero retains
    // the size-derived default. Exhaustion rejects the import; it never
    // materializes a partially solved graph.
    std::size_t type_solver_step_limit = 0;
};

struct PtxImportResult {
    bool ok = false;
    Module module;
    std::vector<std::string> warnings;
    std::vector<std::string> printf_formats;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
    std::string error;
    // Private (st.local) stores whose address cannot be proved exact, each
    // with an 8-byte cell (byte offset in its function's local depot) that
    // holds a pointer and must not be overwritten. Non-empty means the import
    // failed only for want of runtime guards; import_ptx inserts a guard in
    // front of each store that traps if it would overlap any of its cells,
    // and retries.
    struct PrivateStoreGuardRequest {
        int line = 0;
        std::int64_t cell = 0;
        bool operator==(const PrivateStoreGuardRequest&) const = default;
    };
    std::vector<PrivateStoreGuardRequest> private_store_guard_requests;
};

[[nodiscard]] PtxImportResult import_ptx(std::string_view ptx,
                                         const PtxImportOptions& options = {});

}  // namespace cumetal::ir
