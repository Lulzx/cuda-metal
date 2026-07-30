#pragma once

#include "allocation_table.h"
#include "cuda_runtime.h"

#include <functional>

namespace cumetal::rt {

bool resolve_allocation_for_pointer(const void* ptr, AllocationTable::ResolvedAllocation* out);
cudaError_t enqueue_host_operation(cudaStream_t stream, std::function<void()> operation);

}  // namespace cumetal::rt
