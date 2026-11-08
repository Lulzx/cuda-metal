#pragma once

#include "cuda_runtime.h"

// Clean-room declarations for CUDA/OpenGL registration. CuMetal does not map
// OpenGL resources into Metal; these declarations keep programs that also
// provide a non-interop/headless path source-compatible and linkable.
struct cudaGraphicsResource;

enum cudaGraphicsRegisterFlags {
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8,
};

enum cudaGraphicsMapFlags {
    cudaGraphicsMapFlagsNone = cudaGraphicsRegisterFlagsNone,
    cudaGraphicsMapFlagsReadOnly = cudaGraphicsRegisterFlagsReadOnly,
    cudaGraphicsMapFlagsWriteDiscard = cudaGraphicsRegisterFlagsWriteDiscard,
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource,
                                         unsigned int buffer,
                                         unsigned int flags);
cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource** resources,
                                     cudaStream_t stream);
cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource** resources,
                                       cudaStream_t stream);
cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                                 cudaGraphicsResource* resource);
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource* resource);

// Pre-CUDA-3.0 buffer-object interop, deprecated but still called by SiftGPU
// (COLMAP) on its OpenGL path. Like the cudaGraphics* forms above, each one
// refuses with cudaErrorNotSupported; callers' headless paths never reach them.
cudaError_t cudaGLRegisterBufferObject(unsigned int buffer);
cudaError_t cudaGLUnregisterBufferObject(unsigned int buffer);
cudaError_t cudaGLMapBufferObject(void** devPtr, unsigned int buffer);
cudaError_t cudaGLUnmapBufferObject(unsigned int buffer);

#ifdef __cplusplus
}
#endif
