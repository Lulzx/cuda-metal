#pragma once

#include "allocation_table.h"
#include "cuda_runtime.h"
#include "metal_backend.h"

#include <functional>
#include <memory>

namespace cumetal::rt {

bool resolve_allocation_for_pointer(const void* ptr, AllocationTable::ResolvedAllocation* out);
cudaError_t enqueue_host_operation(cudaStream_t stream, std::function<void()> operation);

// The backend stream serving a cudaStream_t. A null handle and the legacy
// handles resolve to the default stream, so a library shim can pass whatever
// stream its handle carries straight through to a backend enqueue.
cudaError_t resolve_backend_stream(cudaStream_t stream,
                                   std::shared_ptr<cumetal::metal_backend::Stream>* out);

// Records `op` as a node on `stream`'s capture graph and returns true, or
// returns false when the stream is not capturing. A library shim calls this
// before doing any work: during capture the work must be recorded rather than
// performed, and the closure has to close over the arguments as they stand now,
// because that is what the graph will replay with.
bool capture_library_call(cudaStream_t stream, std::function<cudaError_t(cudaStream_t)> op);

}  // namespace cumetal::rt
