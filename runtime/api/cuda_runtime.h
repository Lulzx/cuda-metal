#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__clang__) && defined(__CUDA__)
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#ifndef __managed__
#define __managed__ __attribute__((managed))
#endif
#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(t, b) __attribute__((launch_bounds(t, b)))
#endif
#endif

#ifndef __align__
#if defined(__clang__) || defined(__GNUC__)
#define __align__(n) __attribute__((aligned(n)))
#else
#define __align__(n)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchTimeout = 6,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorNotReady = 34,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIllegalAddress = 700,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

typedef struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    constexpr uint3(unsigned int vx = 0, unsigned int vy = 0, unsigned int vz = 0)
        : x(vx), y(vy), z(vz) {}
#endif
} uint3;

typedef struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3(void) const { return uint3{x, y, z}; }
#endif
} dim3;

typedef struct __align__(16) float4 {
    float x;
    float y;
    float z;
    float w;
} float4;

// ── CUDA vector types ────────────────────────────────────────────────────────
// Signed integer vectors
typedef struct { char x, y; }             char2;
typedef struct { char x, y, z; }          char3;
typedef struct { char x, y, z, w; }       char4;
typedef struct { short x, y; }            short2;
typedef struct { short x, y, z; }         short3;
typedef struct { short x, y, z, w; }      short4;
typedef struct { int x, y; }              int2;
typedef struct { int x, y, z; }           int3;
typedef struct __align__(16) { int x, y, z, w; } int4;
typedef struct { long int x, y; }         long2;
typedef struct { long int x, y, z, w; }   long4;
typedef struct { long long int x, y; }    longlong2;
typedef struct { long long int x, y, z, w; } longlong4;
// Unsigned integer vectors
typedef struct { unsigned char x, y; }           uchar2;
typedef struct { unsigned char x, y, z; }         uchar3;
typedef struct { unsigned char x, y, z, w; }      uchar4;
typedef struct { unsigned short x, y; }           ushort2;
typedef struct { unsigned short x, y, z; }        ushort3;
typedef struct { unsigned short x, y, z, w; }     ushort4;
typedef struct { unsigned int x, y; }                        uint2;
typedef struct __align__(16) { unsigned int x, y, z, w; }   uint4;  // uint3 already defined above
typedef struct { unsigned long int x, y; }        ulong2;
typedef struct { unsigned long int x, y, z, w; }  ulong4;
typedef struct { unsigned long long int x, y; }   ulonglong2;
typedef struct { unsigned long long int x, y, z, w; } ulonglong4;
// Floating-point vectors
typedef struct { float x, y; }   float2;
typedef struct { float x, y, z; } float3;
typedef struct { double x, y; }  double2;
typedef struct { double x, y, z; } double3;
typedef struct { double x, y, z, w; } double4;

#ifdef __cplusplus
static inline constexpr char2   make_char2(char x, char y)   { return {x, y}; }
static inline constexpr char4   make_char4(char x, char y, char z, char w) { return {x,y,z,w}; }
static inline constexpr short2  make_short2(short x, short y) { return {x, y}; }
static inline constexpr short4  make_short4(short x, short y, short z, short w) { return {x,y,z,w}; }
static inline constexpr int2    make_int2(int x, int y)       { return {x, y}; }
static inline constexpr int3    make_int3(int x, int y, int z) { return {x, y, z}; }
static inline constexpr int4    make_int4(int x, int y, int z, int w) { return {x,y,z,w}; }
static inline constexpr long2   make_long2(long int x, long int y) { return {x, y}; }
static inline constexpr longlong2 make_longlong2(long long int x, long long int y) { return {x, y}; }
static inline constexpr uchar2  make_uchar2(unsigned char x, unsigned char y) { return {x, y}; }
static inline constexpr uchar4  make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return {x,y,z,w}; }
static inline constexpr ushort2 make_ushort2(unsigned short x, unsigned short y) { return {x, y}; }
static inline constexpr ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { return {x,y,z,w}; }
static inline constexpr uint2   make_uint2(unsigned int x, unsigned int y) { return {x, y}; }
static inline constexpr uint4   make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }
static inline constexpr ulong2  make_ulong2(unsigned long int x, unsigned long int y) { return {x, y}; }
static inline constexpr ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y) { return {x, y}; }
static inline constexpr float2  make_float2(float x, float y) { return {x, y}; }
static inline constexpr float3  make_float3(float x, float y, float z) { return {x, y, z}; }
static inline constexpr float4  make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
static inline constexpr double2 make_double2(double x, double y) { return {x, y}; }
static inline constexpr double4 make_double4(double x, double y, double z, double w) { return {x,y,z,w}; }
#endif

typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int warpSize;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int sharedMemPerBlock;
    int regsPerBlock;
    int major;
    int minor;
    int unifiedAddressing;          // Always 1 on Apple Silicon (UMA)
    int managedMemory;              // Always 1 on Apple Silicon (UMA)
    int concurrentManagedAccess;    // Always 1 on Apple Silicon (UMA)
    int maxBufferArguments;         // 31 (Metal buffer argument limit)
    // Additional fields — populated by cudaGetDeviceProperties (spec §6.8)
    int clockRate;                  // GPU clock in kHz
    int memoryClockRate;            // Memory clock in kHz (same as GPU on UMA)
    int memoryBusWidth;             // Memory bus width in bits
    size_t totalConstMem;           // Constant memory size (64 KB)
    size_t sharedMemPerMultiprocessor; // Shared mem per SM
    int maxThreadsPerMultiProcessor; // Max threads per SM
    int l2CacheSize;                // L2 cache size in bytes
    int canMapHostMemory;           // Always 1 on UMA (host pointers are device pointers)
    int integrated;                 // Always 1 (Apple Silicon is integrated GPU)
    int concurrentKernels;          // 1 (Metal supports concurrent dispatches)
    int asyncEngineCount;           // 0 (UMA makes async memcpy effectively free)
    int computeMode;                // 0 = cudaComputeModeDefault
    int pciBusID;                   // 0 (no PCI on Apple Silicon)
    int pciDeviceID;                // 0
    int pciDomainID;                // 0
    int tccDriver;                  // 0 (not a Tesla compute cluster)
    int kernelExecTimeoutEnabled;   // 0 (Metal does not enforce GPU timeout by default)
    int pageableMemoryAccess;       // 1 (UMA: device can access host pageable memory)
    int pageableMemoryAccessUsesHostPageTables; // 1 (same page tables on Apple Silicon)
} cudaDeviceProp;

typedef enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrConcurrentManagedAccess = 89,
    // Additional attributes corresponding to cudaDeviceProp fields.
    cudaDevAttrMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrPageableMemoryAccess = 92,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 93,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrSharedMemPerBlockOptin = 97,
} cudaDeviceAttr;

typedef enum cudaComputeMode {
    cudaComputeModeDefault         = 0,  // Multiple threads can use device simultaneously
    cudaComputeModeExclusive       = 1,  // Only one thread can use device at a time
    cudaComputeModeProhibited      = 2,  // No thread can use device
    cudaComputeModeExclusiveProcess = 3, // Only one process can use device at a time
} cudaComputeMode;

typedef enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
} cudaFuncCache;

typedef enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
} cudaSharedMemConfig;

typedef struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
} cudaFuncAttributes;

typedef enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
} cudaMemoryType;

typedef struct cudaPointerAttributes {
    cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
} cudaPointerAttributes;

typedef struct cudaStream_st* cudaStream_t;
typedef struct cudaEvent_st* cudaEvent_t;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* user_data);

#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

enum {
    cudaEventDefault = 0x0,
    cudaEventBlockingSync = 0x1,
    cudaEventDisableTiming = 0x2,
    cudaEventInterprocess = 0x4,
};

enum {
    cudaStreamDefault = 0x0,
    cudaStreamNonBlocking = 0x1,
};

enum {
    cudaHostAllocDefault = 0x0,
    cudaHostAllocPortable = 0x1,
    cudaHostAllocMapped = 0x2,
    cudaHostAllocWriteCombined = 0x4,
};

enum {
    cudaDeviceScheduleAuto = 0x00,
    cudaDeviceScheduleSpin = 0x01,
    cudaDeviceScheduleYield = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost = 0x08,
    cudaDeviceLmemResizeToMax = 0x10,
};

typedef enum cumetalArgKind {
    CUMETAL_ARG_BUFFER = 0,
    CUMETAL_ARG_BYTES = 1,
} cumetalArgKind_t;

typedef struct cumetalKernelArgInfo {
    cumetalArgKind_t kind;
    uint32_t size_bytes;
} cumetalKernelArgInfo_t;

typedef struct cumetalKernel {
    const char* metallib_path;
    const char* kernel_name;
    uint32_t arg_count;
    const cumetalKernelArgInfo_t* arg_info;
} cumetalKernel_t;

cudaError_t cudaInit(unsigned int flags);
cudaError_t cudaDriverGetVersion(int* driver_version);
cudaError_t cudaRuntimeGetVersion(int* runtime_version);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device);
cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);
// ── 3D memory types ──────────────────────────────────────────────────────────
typedef struct cudaExtent {
    size_t width;    // Width in bytes (for pitched) or elements (for arrays)
    size_t height;   // Height in elements
    size_t depth;    // Depth in elements
} cudaExtent;

typedef struct cudaPitchedPtr {
    void*  ptr;
    size_t pitch;    // Row pitch in bytes
    size_t xsize;    // Logical width in bytes
    size_t ysize;    // Logical height in elements
} cudaPitchedPtr;

typedef struct cudaPos {
    size_t x;
    size_t y;
    size_t z;
} cudaPos;

// Opaque CUDA array handle (not implemented; present for struct compatibility).
struct cudaArray;
typedef struct cudaArray* cudaArray_t;

typedef struct cudaMemcpy3DParms {
    cudaArray_t      srcArray;
    struct cudaPos   srcPos;
    struct cudaPitchedPtr srcPtr;
    cudaArray_t      dstArray;
    struct cudaPos   dstPos;
    struct cudaPitchedPtr dstPtr;
    struct cudaExtent extent;
    cudaMemcpyKind   kind;
} cudaMemcpy3DParms;

#ifdef __cplusplus
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) {
    cudaExtent e; e.width = w; e.height = h; e.depth = d; return e;
}
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) {
    cudaPos p; p.x = x; p.y = y; p.z = z; return p;
}
static inline cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p,
                                                  size_t xsz, size_t ysz) {
    cudaPitchedPtr pp; pp.ptr = d; pp.pitch = p; pp.xsize = xsz; pp.ysize = ysz; return pp;
}
#endif

cudaError_t cudaMalloc(void** dev_ptr, size_t size);
cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags);
// Pitched 2D allocation — on UMA returns a contiguous allocation with pitch = width rounded
// up to the device's alignment requirement (spec §6.2).
cudaError_t cudaMallocPitch(void** dev_ptr, size_t* pitch, size_t width, size_t height);
// 3D pitched allocation.
cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFree(void* dev_ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream);
cudaError_t cudaMemset(void* dev_ptr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream);
// 2D pitched memcpy — on UMA, copy row-by-row (width bytes per row).
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch,
                          const void* src, size_t spitch,
                          size_t width, size_t height,
                          cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                               const void* src, size_t spitch,
                               size_t width, size_t height,
                               cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset2D(void* dev_ptr, size_t pitch,
                          int value, size_t width, size_t height);
// 3D pitched copy — on UMA, copies plane-by-row.
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p);
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream);
// Unified Memory advisory APIs — no-ops on Apple Silicon UMA (all memory is already managed).
typedef enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6,
} cudaMemoryAdvise;
typedef enum cudaMemRangeAttribute {
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
} cudaMemRangeAttribute;
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream);
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device);
cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute,
                                     const void* devPtr, size_t count);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaLaunchKernel(const void* func,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** args,
                             size_t shared_mem,
                             cudaStream_t stream);
cudaError_t cudaConfigureCall(dim3 grid_dim,
                              dim3 block_dim,
#ifdef __cplusplus
                              size_t shared_mem = 0,
                              cudaStream_t stream = nullptr);
#else
                              size_t shared_mem,
                              cudaStream_t stream);
#endif
cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void* func);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config);
// cudaFuncAttribute: per-function attributes that programs may set.
// Metal has no per-function register limits; accepted as no-ops.
typedef enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
} cudaFuncAttribute;
cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                          const void* func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                    const void* func,
                                                                    int blockSize,
                                                                    size_t dynamicSMemSize,
                                                                    unsigned int flags);
cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize,
                                               int* blockSize,
                                               const void* func,
                                               size_t dynamicSMemSize,
                                               int blockSizeLimit);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop);
// Peer access — Apple Silicon has a single GPU; peer access is unsupported (spec §2.2).
cudaError_t cudaDeviceCanAccessPeer(int* can_access_peer, int device, int peer_device);
cudaError_t cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peer_device);
// Peer memcpy — single GPU on Apple Silicon; peer copies are local copies.
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice,
                            const void* src, int srcDevice,
                            size_t count);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice,
                                 const void* src, int srcDevice,
                                 size_t count, cudaStream_t stream);
// Device-level L1/shared-memory config — no-ops on Metal (no configurable split).
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig);
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig);
// Symbol address/size queries for __device__ variables.
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol);

typedef enum cudaLimit {
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04,
    cudaLimitMaxL2FetchGranularity = 0x05,
    cudaLimitPersistingL2CacheSize = 0x06,
} cudaLimit;

cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
// cudaLaunchHostFunc — enqueues a host callback on the stream (spec §6.9).
typedef void (*cudaHostFn_t)(void* userData);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);

cudaError_t cudaLaunchCooperativeKernel(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && defined(__clang__) && defined(__CUDA__)

#include <limits.h>
#include <math.h>

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>

#if defined(assert)
#undef assert
#define assert(expr) ((void)0)
#endif

static __host__ __forceinline__ int max(int a, int b) {
    return a > b ? a : b;
}

static __host__ __forceinline__ int min(int a, int b) {
    return a < b ? a : b;
}

template <typename T>
static __device__ __forceinline__ T __ldcs(const T* ptr) {
    return *ptr;
}

// __ldg: load via read-only (texture) cache. On UMA Apple Silicon there is no
// dedicated read-only cache, so this is a plain load — identical semantics,
// no performance difference (spec §8).
template <typename T>
static __device__ __forceinline__ T __ldg(const T* ptr) {
    return *ptr;
}

// __ldlu/__ldcv: non-coherent/volatile load hints — plain loads on UMA.
template <typename T>
static __device__ __forceinline__ T __ldlu(const T* ptr) {
    return *ptr;
}

template <typename T>
static __device__ __forceinline__ T __ldcv(const T* ptr) {
    return *ptr;
}

template <typename T>
static __device__ __forceinline__ void __stcs(T* ptr, T value) {
    *ptr = value;
}

// ── Atomic operations ────────────────────────────────────────────────────────

static __device__ __forceinline__ float atomicAdd(float* ptr, float val) {
    return __fAtomicAdd(ptr, val);
}
static __device__ __forceinline__ int atomicAdd(int* ptr, int val) {
    return __iAtomicAdd(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicAdd(unsigned int* ptr, unsigned int val) {
    return __uAtomicAdd(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicAdd(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicAdd(ptr, val);
}
// double atomicAdd via CAS loop (CUDA Volta+ semantics; Apple Silicon UMA has no native FP64 atomic).
static __device__ __forceinline__ double atomicAdd(double* addr, double val) {
    unsigned long long* base = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *base;
    do {
        assumed = old;
        double cur;
        __builtin_memcpy(&cur, &assumed, 8);
        double updated = cur + val;
        unsigned long long updated_bits;
        __builtin_memcpy(&updated_bits, &updated, 8);
        old = __ullAtomicCAS(base, assumed, updated_bits);
    } while (old != assumed);
    double result;
    __builtin_memcpy(&result, &old, 8);
    return result;
}

static __device__ __forceinline__ int atomicSub(int* ptr, int val) {
    return __iAtomicAdd(ptr, -val);
}
static __device__ __forceinline__ unsigned int atomicSub(unsigned int* ptr, unsigned int val) {
    return __uAtomicAdd(ptr, static_cast<unsigned int>(-static_cast<int>(val)));
}

static __device__ __forceinline__ int atomicExch(int* ptr, int val) {
    return __iAtomicExch(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicExch(unsigned int* ptr, unsigned int val) {
    return __uAtomicExch(ptr, val);
}
static __device__ __forceinline__ float atomicExch(float* ptr, float val) {
    return __fAtomicExch(ptr, val);
}

static __device__ __forceinline__ int atomicMin(int* ptr, int val) {
    return __iAtomicMin(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicMin(unsigned int* ptr, unsigned int val) {
    return __uAtomicMin(ptr, val);
}

static __device__ __forceinline__ int atomicMax(int* ptr, int val) {
    return __iAtomicMax(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicMax(unsigned int* ptr, unsigned int val) {
    return __uAtomicMax(ptr, val);
}

static __device__ __forceinline__ unsigned int atomicCAS(unsigned int* ptr,
                                                          unsigned int cmp,
                                                          unsigned int val) {
    return __uAtomicCAS(ptr, cmp, val);
}
static __device__ __forceinline__ int atomicCAS(int* ptr, int cmp, int val) {
    return __iAtomicCAS(ptr, cmp, val);
}
static __device__ __forceinline__ unsigned long long atomicCAS(unsigned long long* ptr,
                                                                unsigned long long cmp,
                                                                unsigned long long val) {
    return __ullAtomicCAS(ptr, cmp, val);
}

static __device__ __forceinline__ int atomicAnd(int* ptr, int val) {
    return __iAtomicAnd(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicAnd(unsigned int* ptr, unsigned int val) {
    return __uAtomicAnd(ptr, val);
}

static __device__ __forceinline__ int atomicOr(int* ptr, int val) {
    return __iAtomicOr(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicOr(unsigned int* ptr, unsigned int val) {
    return __uAtomicOr(ptr, val);
}

static __device__ __forceinline__ int atomicXor(int* ptr, int val) {
    return __iAtomicXor(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicXor(unsigned int* ptr, unsigned int val) {
    return __uAtomicXor(ptr, val);
}

// ── Synchronization, memory fences, bit ops, FMA ────────────────────────────
// Only define when clang's __clang_cuda_device_functions.h hasn't already
// provided these (it uses the guard __CLANG_CUDA_DEVICE_FUNCTIONS_H__).
#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__

static __device__ __forceinline__ void __syncwarp(unsigned int mask = 0xffffffffu) {
    __nvvm_bar_warp_sync(mask);
}

// Warp shuffle intrinsics (spec §5.3).
// On Apple Silicon the full warp participates; partial masks are conservative no-ops.
static __device__ __forceinline__ int __shfl_sync(unsigned int mask, int val, int srcLane, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_idx_i32(val, srcLane, 0x1f);
}
static __device__ __forceinline__ float __shfl_sync(unsigned int mask, float val, int srcLane, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_idx_f32(val, srcLane, 0x1f);
}
static __device__ __forceinline__ int __shfl_down_sync(unsigned int mask, int val, unsigned int delta, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_down_i32(val, delta, 0x1f);
}
static __device__ __forceinline__ float __shfl_down_sync(unsigned int mask, float val, unsigned int delta, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_down_f32(val, delta, 0x1f);
}
static __device__ __forceinline__ int __shfl_up_sync(unsigned int mask, int val, unsigned int delta, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_up_i32(val, delta, 0);
}
static __device__ __forceinline__ float __shfl_up_sync(unsigned int mask, float val, unsigned int delta, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_up_f32(val, delta, 0);
}
static __device__ __forceinline__ int __shfl_xor_sync(unsigned int mask, int val, int laneMask, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_bfly_i32(val, laneMask, 0x1f);
}
static __device__ __forceinline__ float __shfl_xor_sync(unsigned int mask, float val, int laneMask, int width = 32) {
    (void)mask; (void)width;
    return __nvvm_shfl_bfly_f32(val, laneMask, 0x1f);
}

// Warp vote intrinsics (spec §5.3).
static __device__ __forceinline__ int __any_sync(unsigned int mask, int predicate) {
    (void)mask;
    return __nvvm_vote_any(predicate);
}
static __device__ __forceinline__ int __all_sync(unsigned int mask, int predicate) {
    (void)mask;
    return __nvvm_vote_all(predicate);
}
static __device__ __forceinline__ unsigned int __ballot_sync(unsigned int mask, int predicate) {
    (void)mask;
    return __nvvm_vote_ballot(predicate);
}

static __device__ __forceinline__ void __threadfence(void) { __nvvm_membar_gl(); }
static __device__ __forceinline__ void __threadfence_block(void) { __nvvm_membar_cta(); }
static __device__ __forceinline__ void __threadfence_system(void) { __nvvm_membar_sys(); }

static __device__ __forceinline__ unsigned int __activemask(void) {
    return __nvvm_activemask();
}

// Lane mask special registers: bitmasks of lanes with index R relative to current lane.
static __device__ __forceinline__ unsigned int __lanemask_eq(void) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return 1u << laneid;
}
static __device__ __forceinline__ unsigned int __lanemask_lt(void) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return (1u << laneid) - 1u;
}
static __device__ __forceinline__ unsigned int __lanemask_le(void) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return (2u << laneid) - 1u;
}
static __device__ __forceinline__ unsigned int __lanemask_gt(void) {
    return ~__lanemask_le();
}
static __device__ __forceinline__ unsigned int __lanemask_ge(void) {
    return ~__lanemask_lt();
}

// Warp-wide reduction intrinsics (Ampere+, __reduce_*_sync).
// On Apple Silicon, use NVVM simdgroup reduce builtins.
static __device__ __forceinline__ unsigned int __reduce_add_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_add(val);
}
static __device__ __forceinline__ int __reduce_add_sync(unsigned int mask, int val) {
    (void)mask;
    return static_cast<int>(__nvvm_redux_sync_add(static_cast<unsigned>(val)));
}
static __device__ __forceinline__ unsigned int __reduce_and_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_and(val);
}
static __device__ __forceinline__ unsigned int __reduce_or_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_or(val);
}
static __device__ __forceinline__ unsigned int __reduce_xor_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_xor(val);
}
static __device__ __forceinline__ unsigned int __reduce_min_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_umin(val);
}
static __device__ __forceinline__ int __reduce_min_sync(unsigned int mask, int val) {
    (void)mask;
    return __nvvm_redux_sync_min(val);
}
static __device__ __forceinline__ unsigned int __reduce_max_sync(unsigned int mask, unsigned int val) {
    (void)mask;
    return __nvvm_redux_sync_umax(val);
}
static __device__ __forceinline__ int __reduce_max_sync(unsigned int mask, int val) {
    (void)mask;
    return __nvvm_redux_sync_max(val);
}


// Bit manipulation intrinsics
static __device__ __forceinline__ int __popc(unsigned int x) {
    return __builtin_popcount(x);
}
static __device__ __forceinline__ int __popcll(unsigned long long x) {
    return __builtin_popcountll(x);
}
static __device__ __forceinline__ int __clz(int x) {
    return x == 0 ? 32 : __builtin_clz(static_cast<unsigned int>(x));
}
static __device__ __forceinline__ int __clzll(long long x) {
    return x == 0 ? 64 : __builtin_clzll(static_cast<unsigned long long>(x));
}
static __device__ __forceinline__ unsigned int __brev(unsigned int x) {
    return __builtin_bitreverse32(x);
}
static __device__ __forceinline__ unsigned long long __brevll(unsigned long long x) {
    return __builtin_bitreverse64(x);
}
static __device__ __forceinline__ int __ffs(int x) { return __builtin_ffs(x); }
static __device__ __forceinline__ int __ffsll(long long x) { return __builtin_ffsll(x); }

// FMA helpers
static __device__ __forceinline__ float __fmaf_rn(float x, float y, float z) {
    return __builtin_fmaf(x, y, z);
}
static __device__ __forceinline__ double __fma_rn(double x, double y, double z) {
    return __builtin_fma(x, y, z);
}

// Type-punning intrinsics (reinterpret bit pattern, no conversion).
static __device__ __forceinline__ float __int_as_float(int x) {
    float r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}
static __device__ __forceinline__ int __float_as_int(float x) {
    int r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}
static __device__ __forceinline__ float __uint_as_float(unsigned int x) {
    float r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}
static __device__ __forceinline__ unsigned int __float_as_uint(float x) {
    unsigned int r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}
static __device__ __forceinline__ double __longlong_as_double(long long x) {
    double r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}
static __device__ __forceinline__ long long __double_as_longlong(double x) {
    long long r; __builtin_memcpy(&r, &x, sizeof(r)); return r;
}

// Integer device intrinsics.
static __device__ __forceinline__ int __mulhi(int a, int b) {
    return static_cast<int>(static_cast<long long>(a) * b >> 32);
}
static __device__ __forceinline__ unsigned int __umulhi(unsigned int a, unsigned int b) {
    return static_cast<unsigned int>(static_cast<unsigned long long>(a) * b >> 32);
}
static __device__ __forceinline__ int __mul24(int a, int b) {
    return (a & 0xFFFFFF) * (b & 0xFFFFFF);
}
static __device__ __forceinline__ unsigned int __umul24(unsigned int a, unsigned int b) {
    return (a & 0xFFFFFFu) * (b & 0xFFFFFFu);
}
static __device__ __forceinline__ int __sad(int a, int b, int c) {
    return __builtin_abs(a - b) + c;
}
static __device__ __forceinline__ unsigned int __usad(unsigned int a, unsigned int b, unsigned int c) {
    return (a > b ? a - b : b - a) + c;
}

// Fast (reduced-precision) math intrinsics — on Apple Silicon Metal, these map
// directly to the standard FP32 hardware operations (no separate fast-math path).
static __device__ __forceinline__ float __sinf(float x)  { return __builtin_sinf(x); }
static __device__ __forceinline__ float __cosf(float x)  { return __builtin_cosf(x); }
static __device__ __forceinline__ float __tanf(float x)  { return __builtin_tanf(x); }
static __device__ __forceinline__ float __expf(float x)  { return __builtin_expf(x); }
static __device__ __forceinline__ float __exp2f(float x) { return __builtin_exp2f(x); }
static __device__ __forceinline__ float __logf(float x)  { return __builtin_logf(x); }
static __device__ __forceinline__ float __log2f(float x) { return __builtin_log2f(x); }
static __device__ __forceinline__ float __log10f(float x){ return __builtin_log10f(x); }
static __device__ __forceinline__ float __powf(float x, float y) { return __builtin_powf(x, y); }
static __device__ __forceinline__ float __sqrtf(float x) { return __builtin_sqrtf(x); }
static __device__ __forceinline__ float __rsqrtf(float x){ return 1.0f / __builtin_sqrtf(x); }
static __device__ __forceinline__ float __fdividef(float x, float y) { return x / y; }
static __device__ __forceinline__ float __frcp_rn(float x){ return 1.0f / x; }
static __device__ __forceinline__ float __fsqrt_rn(float x){ return __builtin_sqrtf(x); }

#endif  // !__CLANG_CUDA_DEVICE_FUNCTIONS_H__

#endif  // device code section
