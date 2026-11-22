#pragma once
//
// Provenance for library shim calls (cuBLAS, cuBLASLt, cuFFT, cuRAND,
// cuSPARSE, cuSOLVER, cuDNN).
//
// Kernel launches print a CUMETAL_PROVENANCE line under CUMETAL_TRACE_GPU.
// A library call may instead run on MPS, on CuMetal's own library kernels, or
// on the CPU, and an external harness has to be able to tell which. Each
// public entry point opens a LibraryCallScope; the places that actually do
// work note what ran:
//
//   - MPS / MPSGraph dispatch            -> LibraryBackend::kMps
//   - a CuMetal library kernel (incl. VkFFT) -> LibraryBackend::kKernel
//   - a CPU computation: a host operation
//     enqueued on the stream, or an explicit
//     CPU fallback after a GPU attempt     -> LibraryBackend::kHost
//   - the host-library stream wait         -> LibraryBackend::kHostWait
//
// A stream wait precedes both CPU work and some synchronous GPU dispatches
// (cuBLAS SGEMM waits, then encodes MPS), so on its own it only means CPU
// work when no GPU backend ran in the same call.
//
// When the outermost scope closes it prints
//
//   CUMETAL_PROVENANCE event=library_call library=cuBLAS routine=cublasSgemm
//       backend=mps host_work=false
//
// `backend` joins every backend that ran with '+', GPU backends first; a call
// that only waited and computed on the CPU says `host_fallback`. A call that
// did no work (refused, zero-sized, or recorded into a capturing graph)
// prints nothing, except a captured call, which says `backend=graph_capture`.

#include <cstdint>

namespace cumetal::rt {

enum class LibraryBackend : std::uint8_t {
    kHost = 1u << 0,
    kKernel = 1u << 1,
    kMps = 1u << 2,
    kGraphCapture = 1u << 3,
    kHostWait = 1u << 4,
};

class LibraryCallScope {
public:
    LibraryCallScope(const char* library, const char* routine);
    ~LibraryCallScope();
    LibraryCallScope(const LibraryCallScope&) = delete;
    LibraryCallScope& operator=(const LibraryCallScope&) = delete;

private:
    friend void note_library_backend(LibraryBackend backend);
    const char* library_ = nullptr;
    const char* routine_ = nullptr;
    std::uint8_t backends_ = 0;
    bool active_ = false;
};

// Records that `backend` did work for the library call open on this thread.
// No-op when no scope is open or tracing is disabled.
void note_library_backend(LibraryBackend backend);

}  // namespace cumetal::rt

#define CUMETAL_LIBRARY_CALL(library) \
    ::cumetal::rt::LibraryCallScope cumetal_library_call_scope_((library), __func__)
