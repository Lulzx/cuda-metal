#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUdevice;
typedef uint64_t CUdeviceptr;

typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct cudaStream_st* CUstream;
typedef struct cudaEvent_st* CUevent;

#define CU_STREAM_LEGACY ((CUstream)0x1)
#define CU_STREAM_PER_THREAD ((CUstream)0x2)

enum {
    CU_STREAM_DEFAULT = 0x0,
    CU_STREAM_NON_BLOCKING = 0x1,
};

enum {
    CU_CTX_SCHED_AUTO = 0x00,
    CU_CTX_SCHED_SPIN = 0x01,
    CU_CTX_SCHED_YIELD = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_MAP_HOST = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
};

enum {
    CU_MEMHOSTALLOC_PORTABLE = 0x1,
    CU_MEMHOSTALLOC_DEVICEMAP = 0x2,
    CU_MEMHOSTALLOC_WRITECOMBINED = 0x4,
};

#ifndef CU_LAUNCH_PARAM_END
#define CU_LAUNCH_PARAM_END ((void*)0x00)
#endif
#ifndef CU_LAUNCH_PARAM_BUFFER_POINTER
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#endif
#ifndef CU_LAUNCH_PARAM_BUFFER_SIZE
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#endif

typedef enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEVICES_UNAVAILABLE = 46,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_UNKNOWN = 999,
} CUresult;

typedef void (*CUstreamCallback)(CUstream hStream, CUresult status, void* userData);
typedef void (*CUhostFn)(void* userData);

typedef enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
} CUdevice_attribute;

CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);
CUresult cuCtxGetDevice(CUdevice* device);
CUresult cuCtxGetFlags(unsigned int* flags);
CUresult cuCtxSetFlags(unsigned int flags);
CUresult cuCtxSynchronize(void);
// Primary context management.
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult cuDevicePrimaryCtxReset(CUdevice dev);

// Device UUID.
typedef struct { unsigned char bytes[16]; } CUuuid;
CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamGetPriority(CUstream hStream, int* priority);
CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags);
CUresult cuStreamAddCallback(CUstream hStream,
                             CUstreamCallback callback,
                             void* userData,
                             unsigned int flags);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags);
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData);

CUresult cuEventCreate(CUevent* phEvent, unsigned int flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                            CUmodule hmod, const char* name);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module,
                            const void* image,
                            unsigned int numOptions,
                            void* options,
                            void* optionValues);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra);

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST    = 0x01,
    CU_MEMORYTYPE_DEVICE  = 0x02,
    CU_MEMORYTYPE_ARRAY   = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04,
} CUmemorytype;

// Opaque CUDA array handle (not implemented; present for struct compatibility).
typedef void* CUarray;

typedef struct CUDA_MEMCPY3D_st {
    size_t          srcXInBytes;
    size_t          srcY;
    size_t          srcZ;
    size_t          srcLOD;
    CUmemorytype    srcMemoryType;
    const void*     srcHost;
    CUdeviceptr     srcDevice;
    CUarray         srcArray;
    void*           reserved0;
    size_t          srcPitch;
    size_t          srcHeight;

    size_t          dstXInBytes;
    size_t          dstY;
    size_t          dstZ;
    size_t          dstLOD;
    CUmemorytype    dstMemoryType;
    void*           dstHost;
    CUdeviceptr     dstDevice;
    CUarray         dstArray;
    void*           reserved1;
    size_t          dstPitch;
    size_t          dstHeight;

    size_t          WidthInBytes;
    size_t          Height;
    size_t          Depth;
} CUDA_MEMCPY3D;

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemAllocHost(void** pp, size_t bytesize);
CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags);
CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags);
CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p);
CUresult cuMemFreeHost(void* p);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                           const void* srcHost,
                           size_t ByteCount,
                           CUstream hStream);
CUresult cuMemcpyDtoHAsync(void* dstHost,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                           CUdeviceptr srcDevice,
                           size_t ByteCount,
                           CUstream hStream);
CUresult cuMemGetInfo(size_t* freeBytes, size_t* totalBytes);
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);

CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuProfilerStart(void);
CUresult cuProfilerStop(void);

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                    unsigned int gridDimX,
                                    unsigned int gridDimY,
                                    unsigned int gridDimZ,
                                    unsigned int blockDimX,
                                    unsigned int blockDimY,
                                    unsigned int blockDimZ,
                                    unsigned int sharedMemBytes,
                                    CUstream hStream,
                                    void** kernelParams);

CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream);
// Generic device-to-device async copy (infers direction from allocation table).
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream);
// Peer copies — single GPU on Apple Silicon; behave as local D2D copies.
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext,
                      size_t ByteCount);
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                            CUdeviceptr srcDevice, CUcontext srcContext,
                            size_t ByteCount, CUstream hStream);

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                           CUstream hStream);
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
// 2D strided memset (per-row fill with pitch stride).
CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                       unsigned char uc, size_t Width, size_t Height);
CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned short us, size_t Width, size_t Height);
CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                        unsigned int ui, size_t Width, size_t Height);
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned char uc, size_t Width, size_t Height, CUstream hStream);
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                             unsigned short us, size_t Width, size_t Height, CUstream hStream);
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                             unsigned int ui, size_t Width, size_t Height, CUstream hStream);
// Query base address and size of an allocation from the allocation table.
CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT         = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE     = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER  = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER    = 4,
    CU_POINTER_ATTRIBUTE_MAPPED          = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED      = 8,
} CUpointer_attribute;

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);
CUresult cuCtxDisablePeerAccess(CUcontext peerContext);

typedef enum CUfunc_attribute {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
} CUfunc_attribute;

typedef enum CUfunc_cache {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3,
} CUfunc_cache;

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                              CUfunction func,
                                                              int blockSize,
                                                              size_t dynamicSMemSize,
                                                              unsigned int flags);
CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize,
                                          int* blockSize,
                                          CUfunction func,
                                          size_t dynamicSMemSize,
                                          int blockSizeLimit);
CUresult cuFuncGetAttribute(int* pi, CUfunc_attribute attrib, CUfunction hfunc);
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunc_attribute attrib, int value);

#ifdef __cplusplus
}
#endif
