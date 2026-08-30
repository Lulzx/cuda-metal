#pragma once
// Stream ordering for CuMetal's host-backed CUB device-wide algorithms.
//
// These shims run the algorithm on the CPU over unified memory rather than
// launching a Metal kernel. That is a legitimate implementation of the data
// movement -- on Apple Silicon device memory *is* host memory -- but it is not
// automatically a legitimate implementation of the *ordering*. CUDA specifies
// cub::Device* as stream-ordered work: everything already enqueued on the
// stream has completed before the algorithm reads its input.
//
// A host loop has no such ordering. It reads the buffer the moment the call is
// made, which is typically while the producing kernel is still in flight, and
// returns a scan or reduction of whatever happened to be in memory -- usually
// zeros. No error is raised; the answer is simply wrong. GROMACS's nbnxm
// pair-list sort hit exactly this: the prune kernel fills sciHistogram, the
// exclusive scan of it ran before the kernel did, and the resulting sorted
// pair list dropped about a quarter of the interactions with no diagnostic.
//
// Every host-backed entry point therefore synchronizes its stream first.

#include <cuda_runtime.h>

namespace cub {
namespace detail {

// Waits for work already enqueued on `stream` (the null stream being the
// default stream) so the host may read what that work produced.
inline cudaError_t sync_host_backed(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

}  // namespace detail
}  // namespace cub
