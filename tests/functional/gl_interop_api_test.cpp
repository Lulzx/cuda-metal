// OpenGL interop is not implemented: every entry point, including the
// pre-CUDA-3.0 cudaGL*BufferObject forms SiftGPU calls, must refuse with
// cudaErrorNotSupported (never report success without doing the work), leave
// an output pointer null, and record the error for cudaGetLastError.
#include "cuda_gl_interop.h"

#include <cstdio>

namespace {

int failures = 0;

void expect_not_supported(cudaError_t status, const char* what) {
    if (status != cudaErrorNotSupported) {
        std::fprintf(stderr, "FAIL: %s returned %d, want cudaErrorNotSupported\n", what,
                     static_cast<int>(status));
        ++failures;
    }
    const cudaError_t last = cudaGetLastError();
    if (last != cudaErrorNotSupported) {
        std::fprintf(stderr, "FAIL: %s left cudaGetLastError=%d\n", what, static_cast<int>(last));
        ++failures;
    }
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: runtime init\n");
        return 1;
    }

    expect_not_supported(cudaGLRegisterBufferObject(7), "cudaGLRegisterBufferObject");
    void* mapped = reinterpret_cast<void*>(0x1);
    expect_not_supported(cudaGLMapBufferObject(&mapped, 7), "cudaGLMapBufferObject");
    if (mapped != nullptr) {
        std::fprintf(stderr, "FAIL: cudaGLMapBufferObject left a non-null pointer\n");
        ++failures;
    }
    expect_not_supported(cudaGLMapBufferObject(nullptr, 7), "cudaGLMapBufferObject(null)");
    expect_not_supported(cudaGLUnmapBufferObject(7), "cudaGLUnmapBufferObject");
    expect_not_supported(cudaGLUnregisterBufferObject(7), "cudaGLUnregisterBufferObject");

    cudaGraphicsResource* resource = nullptr;
    expect_not_supported(cudaGraphicsGLRegisterBuffer(&resource, 7, cudaGraphicsRegisterFlagsNone),
                         "cudaGraphicsGLRegisterBuffer");
    expect_not_supported(cudaGraphicsMapResources(1, &resource, nullptr), "cudaGraphicsMapResources");
    void* pointer = nullptr;
    size_t size = 0;
    expect_not_supported(cudaGraphicsResourceGetMappedPointer(&pointer, &size, resource),
                         "cudaGraphicsResourceGetMappedPointer");
    expect_not_supported(cudaGraphicsUnmapResources(1, &resource, nullptr), "cudaGraphicsUnmapResources");
    expect_not_supported(cudaGraphicsUnregisterResource(resource), "cudaGraphicsUnregisterResource");

    if (failures != 0) return 1;
    std::printf("PASS: OpenGL interop refuses with cudaErrorNotSupported\n");
    return 0;
}
