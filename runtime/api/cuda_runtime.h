#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __CUDA_RUNTIME_H__
#define __CUDA_RUNTIME_H__ 1
#endif
// cuda_runtime.h is CuMetal's umbrella header: the headers below are pure
// forwarders back to it, so anything that includes this one has already got
// their contents and must see their guards defined too. helper_cuda.h keys
// checkCudaErrors()/_cudaGetErrorEnum() off __DRIVER_TYPES_H__ specifically.
#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__ 1
#endif
#ifndef __CUDA_RUNTIME_API_H__
#define __CUDA_RUNTIME_API_H__ 1
#endif
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__ 1
#endif
#ifndef __VECTOR_FUNCTIONS_H__
#define __VECTOR_FUNCTIONS_H__ 1
#endif
#ifndef __CHANNEL_DESCRIPTOR_H__
#define __CHANNEL_DESCRIPTOR_H__ 1
#endif
#ifndef __DEVICE_LAUNCH_PARAMETERS_H__
#define __DEVICE_LAUNCH_PARAMETERS_H__ 1
#endif
#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__ 1
#endif
#ifndef __CUDA_PROFILER_API_H__
#define __CUDA_PROFILER_API_H__ 1
#endif


#include <stddef.h>
#include <stdint.h>
// Host samples often omit <stdlib.h> because nvcc's CUDA headers transitively
// declare malloc/free/exit. cumetalc force-includes this header, so pull those
// declarations in here for source compatibility.
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

#ifndef CUDARTAPI
#define CUDARTAPI
#endif

// CUDART_CB — calling convention for host callbacks (__stdcall on Windows, empty
// elsewhere). Samples spell callbacks `void CUDART_CB fn(void *)`; with the macro
// undeclared that parses as a variable of type void.
#ifndef CUDART_CB
#define CUDART_CB
#endif

// __grid_constant__ marks a by-value kernel parameter as read-only and uniform
// across the grid. CuMetal already copies kernel arguments into Metal's argument
// buffer, so the annotation carries no codegen difference here.
#ifndef __grid_constant__
#define __grid_constant__
#endif

#if defined(__clang__) && defined(__CUDA__)
#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDA_VERSION
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
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#endif
#ifndef __builtin_align__
#define __builtin_align__(n) __attribute__((aligned(n)))
#endif
#else
// CUDA's public headers keep shared host/device declarations parseable when
// included by an ordinary C++ compiler.  Projects such as PhysX rely on this
// for headers containing __host__ __device__ helpers.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __managed__
#define __managed__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __builtin_align__
#define __builtin_align__(n) __attribute__((aligned(n)))
#endif
#ifndef __device_builtin__
#define __device_builtin__
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

// Clang's CUDA device frontend needs a device declaration in scope before a
// kernel can call printf. The host libc declaration remains host-only; CUDA
// overload resolution selects this declaration in device code, which Clang
// lowers to PTX vprintf for CuMetal's bounded ring-buffer backend.
#if defined(__clang__) && defined(__CUDA__)
__device__ int printf(const char* format, ...);
// Clang's CUDA <new> wrapper implements device new/delete in terms of these
// C allocation entry points. The definitions are supplied by CuMetal's device
// heap lowering rather than the host libc symbols with the same signatures.
__device__ void* malloc(size_t size);
__device__ void free(void* pointer);
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchTimeout = 6,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorNotReady = 34,
    cudaErrorPeerAccessAlreadyEnabled = 50,
    cudaErrorPeerAccessNotEnabled = 51,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorUnknown = 999,
    // Deprecated numbering CUDA keeps for source compatibility. cudaErrorAssert
    // is what a device-side assert() reports; samples compare against it by name.
    cudaErrorAssert = 710,
    cudaErrorInvalidDeviceFunction = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidDevice = 10,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInsufficientDriver = 35,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorCudartUnloading = 4,
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

#ifndef CUMETAL_CUDA_VECTOR_TYPES_DEFINED
#define CUMETAL_CUDA_VECTOR_TYPES_DEFINED 1
#endif

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
static inline constexpr uint3   make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x, y, z}; }
static inline constexpr uint4   make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }
static inline constexpr ulong2  make_ulong2(unsigned long int x, unsigned long int y) { return {x, y}; }
static inline constexpr ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y) { return {x, y}; }
static inline constexpr float2  make_float2(float x, float y) { return {x, y}; }
static inline constexpr float3  make_float3(float x, float y, float z) { return {x, y, z}; }
static inline constexpr float4  make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }
static inline constexpr double2 make_double2(double x, double y) { return {x, y}; }
static inline constexpr double4 make_double4(double x, double y, double z, double w) { return {x,y,z,w}; }
#endif

typedef struct cudaUUID_t { unsigned char bytes[16]; } cudaUUID_t;

typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int warpSize;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int sharedMemPerBlock;
    size_t sharedMemPerBlockOptin;
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
    int pageableMemoryAccess;       // 0 (arbitrary malloc pointers are not Metal-bound)
    int pageableMemoryAccessUsesHostPageTables; // 0 (CuMetal requires tracked allocations)
    int cooperativeLaunch;          // 1 for resident-grid barrier emulation
    int cooperativeMultiDeviceLaunch; // 0 (single device)
    // Growing this struct breaks any consumer binary built against an older
    // header: callers stack-allocate a cudaDeviceProp and cudaGetDeviceProperties
    // writes past the end of the smaller frame. Real CUDA absorbs new fields into
    // fixed-size reserved space; do the same so the next addition is free. Take a
    // slot from here rather than appending, and rebuild consumers if you cannot.
    int persistingL2CacheMaxSize;    // accepted performance-hint budget
    int accessPolicyMaxWindowSize;   // accepted stream access-window budget
    int ECCEnabled;                  // 0 (Apple Silicon unified memory has no ECC)
    cudaUUID_t uuid;                 // Stable CuMetal device identity
    int cumetalReserved[55];
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
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrSharedMemPerBlockOptin = 97,
    cudaDevAttrMemoryPoolsSupported = 115,
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

#ifndef CUMETAL_CUDA_STREAM_T_DEFINED
#define CUMETAL_CUDA_STREAM_T_DEFINED 1
typedef struct CUstream_st* cudaStream_t;
#endif  // CUMETAL_CUDA_STREAM_T_DEFINED
typedef struct CUevent_st* cudaEvent_t;
#if defined(__clang__) && defined(__CUDA__)
// Clang resolves device-side `<<<...>>>` syntax through this declaration when
// relocatable device code is enabled. CuMetal lowers the resulting PTX call to
// its device-launch queue rather than linking NVIDIA's cudadevrt.
__device__ void* cudaGetParameterBuffer(size_t alignment, size_t size);
__device__ cudaError_t cudaLaunchDevice(void*, void**, dim3, dim3,
                                        unsigned int, cudaStream_t);
#endif
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* user_data);

typedef enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2,
} cudaAccessProperty;

typedef struct cudaAccessPolicyWindow {
    void* base_ptr;
    size_t num_bytes;
    float hitRatio;
    cudaAccessProperty hitProp;
    cudaAccessProperty missProp;
} cudaAccessPolicyWindow;

typedef enum cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow = 1,
} cudaStreamAttrID;

typedef union cudaStreamAttrValue {
    cudaAccessPolicyWindow accessPolicyWindow;
} cudaStreamAttrValue;

#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

enum {
    cudaEventDefault = 0x0,
    cudaEventBlockingSync = 0x1,
    cudaEventDisableTiming = 0x2,
    cudaEventInterprocess = 0x4,
};

// Flags for cudaEventRecordWithFlags / cudaStreamWaitEvent. The External
// variants only mean anything inside a stream capture, where they mark an
// event as crossing the captured graph's boundary. CuMetal records and waits
// the same way either way; the names exist because GROMACS passes them
// unconditionally.
enum {
    cudaEventRecordDefault = 0x0,
    cudaEventRecordExternal = 0x1,
};

enum {
    cudaEventWaitDefault = 0x0,
    cudaEventWaitExternal = 0x1,
};

// cudaMallocManaged attachment flags.
enum {
    cudaMemAttachGlobal = 0x01,
    cudaMemAttachHost = 0x02,
    cudaMemAttachSingle = 0x04,
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
    cudaHostRegisterDefault = 0x0,
    cudaHostRegisterPortable = 0x1,
    cudaHostRegisterMapped = 0x2,
    cudaHostRegisterIoMemory = 0x4,
    cudaHostRegisterReadOnly = 0x8,
};

typedef enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
} cudaStreamCaptureMode;

typedef enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
} cudaStreamCaptureStatus;

// ── CUDA Graphs ──────────────────────────────────────────────────────────────
typedef struct cudaGraph_st* cudaGraph_t;
typedef struct cudaGraphExec_st* cudaGraphExec_t;
typedef struct cudaGraphNode_st* cudaGraphNode_t;

typedef enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0x00,
    cudaGraphNodeTypeMemcpy = 0x01,
    cudaGraphNodeTypeMemset = 0x02,
    cudaGraphNodeTypeHost = 0x03,
    cudaGraphNodeTypeGraph = 0x04,
    cudaGraphNodeTypeEmpty = 0x05,
    cudaGraphNodeTypeWaitEvent = 0x06,
    cudaGraphNodeTypeEventRecord = 0x07,
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08,
    cudaGraphNodeTypeExtSemaphoreWait = 0x09,
    cudaGraphNodeTypeMemAlloc = 0x0a,
    cudaGraphNodeTypeMemFree = 0x0b,
    cudaGraphNodeTypeConditional = 0x0d,
    cudaGraphNodeTypeReserved16 = 0x10,
    cudaGraphNodeTypeCount,
} cudaGraphNodeType;

typedef enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent = 0x0,
    cudaGraphMemAttrUsedMemHigh = 0x1,
    cudaGraphMemAttrReservedMemCurrent = 0x2,
    cudaGraphMemAttrReservedMemHigh = 0x3,
} cudaGraphMemAttributeType;

typedef enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUpload = 2,
    cudaGraphInstantiateFlagDeviceLaunch = 4,
    cudaGraphInstantiateFlagUseNodePriority = 8,
} cudaGraphInstantiateFlags;

typedef enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
    cudaGraphExecUpdateErrorAttributesChanged = 8,
} cudaGraphExecUpdateResult;

// CUDA 12 replaced cudaGraphExecUpdate's out-parameter with this struct.
// GROMACS's mdgraph_gpu_impl.cu reads .result to tell a topology change (which
// it recovers from by re-instantiating) from a real failure.
typedef struct cudaGraphExecUpdateResultInfo {
    cudaGraphExecUpdateResult result;
    cudaGraphNode_t errorNode;
    cudaGraphNode_t errorFromNode;
} cudaGraphExecUpdateResultInfo;

enum {
    cudaDeviceScheduleAuto = 0x00,
    cudaDeviceScheduleSpin = 0x01,
    cudaDeviceScheduleYield = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost = 0x08,
    cudaDeviceLmemResizeToMax = 0x10,
    cudaDeviceScheduleMask = 0x07,
    cudaDeviceMask = 0x1f,
    // Deprecated spellings kept by real CUDA for source compatibility.
    cudaDeviceBlockingSync = 0x04,
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

// Opaque CUDA array handle.
struct cudaArray;
typedef struct cudaArray* cudaArray_t;
typedef const struct cudaArray* cudaArray_const_t;

// ── Texture / Surface objects ────────────────────────────────────────────────
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;

typedef enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
} cudaChannelFormatKind;

typedef struct cudaChannelFormatDesc {
    int x, y, z, w;
    cudaChannelFormatKind f;
} cudaChannelFormatDesc;

enum {
    cudaArrayDefault = 0,
    cudaArrayLayered = 0x01,
    cudaArraySurfaceLoadStore = 0x02,
    cudaArrayCubemap = 0x04,
};

typedef enum cudaSurfaceBoundaryMode {
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2,
} cudaSurfaceBoundaryMode;

typedef enum cudaTextureAddressMode {
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3,
} cudaTextureAddressMode;

typedef enum cudaTextureFilterMode {
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1,
} cudaTextureFilterMode;

typedef enum cudaTextureReadMode {
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1,
} cudaTextureReadMode;

typedef struct cudaResourceDesc {
    enum { cudaResourceTypeArray = 0, cudaResourceTypeMipmappedArray = 1,
           cudaResourceTypeLinear = 2, cudaResourceTypePitch2D = 3 } resType;
    union {
        struct { cudaArray_t array; } array;
        struct { void* devPtr; cudaChannelFormatDesc desc; size_t sizeInBytes; } linear;
        struct { void* devPtr; cudaChannelFormatDesc desc; size_t width; size_t height;
                 size_t pitchInBytes; } pitch2D;
    } res;
} cudaResourceDesc;

#ifdef __cplusplus
// The CUDA Toolkit declares the resource-type constants as `enum cudaResourceType`
// at namespace scope, and ordinary CUDA code spells them unqualified. CuMetal grew
// them as an anonymous enum nested in cudaResourceDesc, so stock sources failed to
// compile until they qualified every use. Re-export both spellings.
typedef decltype(cudaResourceDesc::cudaResourceTypeArray) cudaResourceType;

constexpr cudaResourceType cudaResourceTypeArray = cudaResourceDesc::cudaResourceTypeArray;
constexpr cudaResourceType cudaResourceTypeMipmappedArray = cudaResourceDesc::cudaResourceTypeMipmappedArray;
constexpr cudaResourceType cudaResourceTypeLinear = cudaResourceDesc::cudaResourceTypeLinear;
constexpr cudaResourceType cudaResourceTypePitch2D = cudaResourceDesc::cudaResourceTypePitch2D;
#endif  // __cplusplus

typedef struct cudaTextureDesc {
    cudaTextureAddressMode addressMode[3];
    cudaTextureFilterMode filterMode;
    cudaTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    int disableTrilinearOptimization;
} cudaTextureDesc;

typedef struct cudaResourceViewDesc {
    enum { cudaResViewFormatNone = 0 } format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
} cudaResourceViewDesc;

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

typedef struct cudaMemcpy3DPeerParms {
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;
    struct cudaExtent extent;
} cudaMemcpy3DPeerParms;

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
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFree(void* dev_ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
__host__ __device__ cudaError_t cudaMemcpyAsync(void* dst,
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
cudaError_t cudaMemset2DAsync(void* dev_ptr, size_t pitch,
                               int value, size_t width, size_t height,
                               cudaStream_t stream);
// 3D pitched fill — fills each row of each plane with value.
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);
cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent,
                               cudaStream_t stream);
// 3D pitched copy — on UMA, copies plane-by-row.
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p);
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream);
cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream);
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
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length,
                                     unsigned int flags);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
__host__ __device__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags, int priority);
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags);
cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue* value);
cudaError_t cudaStreamGetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   cudaStreamAttrValue* value);
cudaError_t cudaCtxResetPersistingL2Cache(void);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
__host__ __device__ cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus);
cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                  cudaGraphNode_t* pErrorNode, char* pLogBuffer,
                                  size_t bufferSize);
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                           cudaGraph_t graph,
                                           unsigned long long flags);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphNode_t* hErrorNode_out,
                                cudaGraphExecUpdateResult* updateResult_out);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes);
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
__host__ __device__ cudaError_t cudaDeviceSynchronize(void);
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
__host__ __device__ cudaError_t cudaGetLastError(void);
__host__ __device__ cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
__host__ __device__ const char* cudaGetErrorString(cudaError_t error);
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

// ── Texture / Surface API ────────────────────────────────────────────────────
cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                             size_t width, size_t height, unsigned int flags);
cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                              cudaExtent extent, unsigned int flags);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                 const void* src, size_t spitch, size_t width,
                                 size_t height, cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src,
                                   size_t wOffset, size_t hOffset, size_t width,
                                   size_t height, cudaMemcpyKind kind);
cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                               const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset,
                                 size_t hOffset, size_t count, cudaMemcpyKind kind);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                                     const cudaResourceDesc* pResDesc,
                                     const cudaTextureDesc* pTexDesc,
                                     const cudaResourceViewDesc* pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc,
                                              cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc,
                                             cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc,
                                                  cudaTextureObject_t texObject);
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                                     const cudaResourceDesc* pResDesc);
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc,
                                              cudaSurfaceObject_t surfObject);
cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                             cudaChannelFormatKind f);

// Async memory pool API — allocation is host-side on UMA, while frees and
// lifetime transitions remain ordered by the selected CUDA stream.
typedef struct cudaMemPool_st* cudaMemPool_t;

typedef enum cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0,
    cudaMemAllocationTypePinned = 1,
    cudaMemAllocationTypeManaged = 2,
} cudaMemAllocationType;

typedef enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
} cudaMemAllocationHandleType;

typedef enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
} cudaMemLocationType;

typedef struct cudaMemLocation {
    cudaMemLocationType type;
    int id;
} cudaMemLocation;

typedef enum cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone = 0,
    cudaMemAccessFlagsProtRead = 1,
    cudaMemAccessFlagsProtReadWrite = 3,
} cudaMemAccessFlags;

typedef struct cudaMemAccessDesc {
    cudaMemLocation location;
    cudaMemAccessFlags flags;
} cudaMemAccessDesc;

typedef enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowOpportunistic = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold = 4,
    cudaMemPoolAttrReservedMemCurrent = 5,
    cudaMemPoolAttrReservedMemHigh = 6,
    cudaMemPoolAttrUsedMemCurrent = 7,
    cudaMemPoolAttrUsedMemHigh = 8,
} cudaMemPoolAttr;

typedef struct cudaMemPoolProps {
    cudaMemAllocationType allocType;
    cudaMemAllocationHandleType handleTypes;
    cudaMemLocation location;
    void* win32SecurityAttributes;
    unsigned char reserved[64];
} cudaMemPoolProps;

typedef struct cudaMemAllocNodeParams {
    cudaMemPoolProps poolProps;
    cudaMemAccessDesc* accessDescs;
    size_t accessDescCount;
    size_t bytesize;
    void* dptr;
} cudaMemAllocNodeParams;

cudaError_t cudaMallocAsync(void** dev_ptr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void* dev_ptr, cudaStream_t stream);
cudaError_t cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps* poolProps);
cudaError_t cudaMemPoolDestroy(cudaMemPool_t pool);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void* value);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device);
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t pool);
cudaError_t cudaMallocFromPoolAsync(void** dev_ptr, size_t size, cudaMemPool_t pool, cudaStream_t stream);

// Graph node addition APIs
typedef struct cudaKernelNodeParams {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** kernelParams;
    unsigned int sharedMemBytes;
    void* extra;
} cudaKernelNodeParams;

typedef struct cudaMemsetParams {
    void* dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
} cudaMemsetParams;

typedef struct cudaHostNodeParams {
    cudaHostFn_t fn;
    void* userData;
} cudaHostNodeParams;

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaMemcpy3DParms* pCopyParams);
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode,
                                      cudaGraph_t graph,
                                      const cudaGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      void* dst,
                                      const void* src,
                                      size_t count,
                                      cudaMemcpyKind kind);
cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaMemsetParams* pMemsetParams);
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                  const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                  const cudaHostNodeParams* pNodeParams);
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode,
                                     cudaGraph_t graph,
                                     const cudaGraphNode_t* pDependencies,
                                     size_t numDependencies,
                                     cudaMemAllocNodeParams* nodeParams);
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode,
                                    cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies,
                                    size_t numDependencies,
                                    void* dptr);
cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                                           cudaMemAllocNodeParams* params_out);
cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void** dptr_out);
cudaError_t cudaDeviceGetGraphMemAttribute(int device,
                                            cudaGraphMemAttributeType attr,
                                            void* value);
cudaError_t cudaDeviceSetGraphMemAttribute(int device,
                                            cudaGraphMemAttributeType attr,
                                            void* value);
cudaError_t cudaDeviceGraphMemTrim(int device);
cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                                          const cudaKernelNodeParams* nodeParams);
cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                                          const cudaMemcpy3DParms* params);
cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst,
                                            const void* src, size_t count,
                                            cudaMemcpyKind kind);
cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                                          const cudaMemsetParams* params);
cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                                        const cudaHostNodeParams* params);
cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec,
                                              cudaGraphNode_t hNode,
                                              const cudaKernelNodeParams* nodeParams);
cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t graphExec,
                                              cudaGraphNode_t node,
                                              const cudaMemcpy3DParms* params);
cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t graphExec,
                                                cudaGraphNode_t node, void* dst,
                                                const void* src, size_t count,
                                                cudaMemcpyKind kind);
cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t graphExec,
                                              cudaGraphNode_t node,
                                              const cudaMemsetParams* params);
cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t graphExec,
                                            cudaGraphNode_t node,
                                            const cudaHostNodeParams* params);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus,
                                      unsigned long long* pId);

// Legacy thread API — deprecated aliases retained for source compatibility.
cudaError_t cudaThreadExit(void);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig);
cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// CUDA headers provide C++ overloads for kernel-function pointers on several
// runtime APIs. CuMetal's core C ABI uses `const void *`; these wrappers keep
// source compatibility with code that passes typed kernel pointers directly.
#include <type_traits>

// Two-argument cudaEventCreate. In the real CUDA headers this is a C++-only
// overload sitting outside the extern "C" block, not a separate entry point;
// it forwards to cudaEventCreateWithFlags. GROMACS's gpuregiontimer.cuh calls
// it to build timing events with cudaEventDefault.
static inline cudaError_t cudaEventCreate(cudaEvent_t* event, unsigned int flags) {
    return cudaEventCreateWithFlags(event, flags);
}

// CUDA 12 forms of the graph instantiate/update calls. The 5-argument
// cudaGraphInstantiate and the cudaGraphExecUpdateResult* cudaGraphExecUpdate
// above are the CUDA 11 spellings; both are still in use, so both are offered.
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                               cudaGraph_t graph,
                                               unsigned long long flags) {
    return cudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
}

static inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                              cudaGraph_t hGraph,
                                              cudaGraphExecUpdateResultInfo* resultInfo_out) {
    cudaGraphNode_t error_node = nullptr;
    cudaGraphExecUpdateResult result = cudaGraphExecUpdateSuccess;
    const cudaError_t status = cudaGraphExecUpdate(hGraphExec, hGraph, &error_node, &result);
    if (resultInfo_out != nullptr) {
        resultInfo_out->result = result;
        resultInfo_out->errorNode = error_node;
        resultInfo_out->errorFromNode = nullptr;
    }
    // CUDA 12 reports an unsuccessful update through the return code as well as
    // the struct; the CUDA 11 entry point only sets the out-parameter. GROMACS
    // keys its re-instantiate path on the return code, so translate.
    if (status == cudaSuccess && result != cudaGraphExecUpdateSuccess) {
        return cudaErrorGraphExecUpdateFailure;
    }
    return status;
}

struct __cumetal_texture_descriptor {
    unsigned long long data;
    unsigned long long width;
    unsigned long long height;
    unsigned long long depth;
    unsigned long long pitch_bytes;
    unsigned int element_bytes;
    unsigned int channel_kind;
    unsigned int read_mode;
    unsigned int filter_mode;
    unsigned int normalized_coords;
    unsigned int address_mode[3];
};

template <typename T>
static inline cudaChannelFormatDesc cudaCreateChannelDesc() {
    cudaChannelFormatDesc desc{};
    desc.x = static_cast<int>(sizeof(T) * 8);
    desc.f = std::is_floating_point<T>::value
                 ? cudaChannelFormatKindFloat
                 : (std::is_signed<T>::value ? cudaChannelFormatKindSigned
                                             : cudaChannelFormatKindUnsigned);
    return desc;
}

static inline cudaError_t cudaMallocArray(cudaArray_t* array,
                                          const cudaChannelFormatDesc* desc,
                                          size_t width, size_t height = 0) {
    return ::cudaMallocArray(array, desc, width, height, cudaArrayDefault);
}

static inline cudaError_t cudaMalloc3DArray(cudaArray_t* array,
                                            const cudaChannelFormatDesc* desc,
                                            cudaExtent extent) {
    return ::cudaMalloc3DArray(array, desc, extent, cudaArrayDefault);
}

__device__ __forceinline__ int __cumetal_texture_index(float coordinate,
                                                        unsigned long long size,
                                                        unsigned int normalized,
                                                        unsigned int address_mode) {
    float scaled = normalized ? coordinate * static_cast<float>(size) : coordinate;
    int index = static_cast<int>(scaled);
    if (static_cast<float>(index) > scaled) --index;
    const int extent = static_cast<int>(size);
    if (address_mode == cudaAddressModeWrap && extent > 0) {
        index %= extent;
        if (index < 0) index += extent;
    } else if (address_mode == cudaAddressModeMirror && extent > 0) {
        const int period = extent * 2;
        index %= period;
        if (index < 0) index += period;
        if (index >= extent) index = period - 1 - index;
    } else {
        if (index < 0) index = 0;
        if (index >= extent) index = extent - 1;
    }
    return index;
}

template <typename T>
__device__ __forceinline__ T __cumetal_texture_load(
    const __cumetal_texture_descriptor* descriptor, int x, int y, int z) {
    const unsigned long long offset =
        static_cast<unsigned long long>(z) * descriptor->pitch_bytes * descriptor->height +
        static_cast<unsigned long long>(y) * descriptor->pitch_bytes +
        static_cast<unsigned long long>(x) * descriptor->element_bytes;
    const unsigned char* bytes =
        reinterpret_cast<const unsigned char*>(descriptor->data + offset);
    if constexpr (std::is_arithmetic<T>::value) {
        if (descriptor->read_mode == cudaReadModeNormalizedFloat &&
            descriptor->channel_kind == cudaChannelFormatKindUnsigned &&
            descriptor->element_bytes == 1) {
            return static_cast<T>(static_cast<float>(*bytes) / 255.0f);
        }
    }
    return *reinterpret_cast<const T*>(bytes);
}

template <typename T>
__device__ __forceinline__ T tex1Dfetch(cudaTextureObject_t texture, int x) {
    const auto* descriptor = reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
    const int extent = static_cast<int>(descriptor->width);
    if (descriptor->address_mode[0] == cudaAddressModeBorder &&
        (x < 0 || x >= extent)) {
        return T{};
    }
    const int ix = x < 0 ? 0 : (x >= extent ? extent - 1 : x);
    return __cumetal_texture_load<T>(descriptor, ix, 0, 0);
}

template <typename T>
__device__ __forceinline__ T tex2D(cudaTextureObject_t texture, float x, float y) {
    const auto* descriptor = reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
    if (descriptor->filter_mode == cudaFilterModeLinear) {
        const float sx = (descriptor->normalized_coords
                              ? x * static_cast<float>(descriptor->width) : x) - 0.5f;
        const float sy = (descriptor->normalized_coords
                              ? y * static_cast<float>(descriptor->height) : y) - 0.5f;
        int x0 = static_cast<int>(sx); if (static_cast<float>(x0) > sx) --x0;
        int y0 = static_cast<int>(sy); if (static_cast<float>(y0) > sy) --y0;
        const float fx = sx - static_cast<float>(x0);
        const float fy = sy - static_cast<float>(y0);
        const unsigned int address_x = descriptor->normalized_coords
                                           ? descriptor->address_mode[0]
                                           : cudaAddressModeClamp;
        const unsigned int address_y = descriptor->normalized_coords
                                           ? descriptor->address_mode[1]
                                           : cudaAddressModeClamp;
        const int ix0 = __cumetal_texture_index(static_cast<float>(x0), descriptor->width,
                                                0, address_x);
        const int ix1 = __cumetal_texture_index(static_cast<float>(x0 + 1), descriptor->width,
                                                0, address_x);
        const int iy0 = __cumetal_texture_index(static_cast<float>(y0), descriptor->height,
                                                0, address_y);
        const int iy1 = __cumetal_texture_index(static_cast<float>(y0 + 1), descriptor->height,
                                                0, address_y);
        const T p00 = __cumetal_texture_load<T>(descriptor, ix0, iy0, 0);
        const T p10 = __cumetal_texture_load<T>(descriptor, ix1, iy0, 0);
        const T p01 = __cumetal_texture_load<T>(descriptor, ix0, iy1, 0);
        const T p11 = __cumetal_texture_load<T>(descriptor, ix1, iy1, 0);
        return static_cast<T>((1.0f - fy) * ((1.0f - fx) * p00 + fx * p10) +
                              fy * ((1.0f - fx) * p01 + fx * p11));
    }
    const int ix = __cumetal_texture_index(x, descriptor->width,
                                           descriptor->normalized_coords,
                                           descriptor->normalized_coords
                                               ? descriptor->address_mode[0]
                                               : cudaAddressModeClamp);
    const int iy = __cumetal_texture_index(y, descriptor->height,
                                           descriptor->normalized_coords,
                                           descriptor->normalized_coords
                                               ? descriptor->address_mode[1]
                                               : cudaAddressModeClamp);
    return __cumetal_texture_load<T>(descriptor, ix, iy, 0);
}

template <typename T>
__device__ __forceinline__ T tex2DLayered(cudaTextureObject_t texture,
                                          float x, float y, int layer) {
    const auto* descriptor = reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
    const int ix = __cumetal_texture_index(x, descriptor->width,
                                           descriptor->normalized_coords,
                                           descriptor->address_mode[0]);
    const int iy = __cumetal_texture_index(y, descriptor->height,
                                           descriptor->normalized_coords,
                                           descriptor->address_mode[1]);
    const int iz = layer < 0 ? 0 : (layer >= static_cast<int>(descriptor->depth)
                                        ? static_cast<int>(descriptor->depth) - 1 : layer);
    return __cumetal_texture_load<T>(descriptor, ix, iy, iz);
}

template <typename T>
__device__ __forceinline__ T tex3D(cudaTextureObject_t texture,
                                   float x, float y, float z) {
    const auto* descriptor = reinterpret_cast<const __cumetal_texture_descriptor*>(texture);
    const int ix = __cumetal_texture_index(x, descriptor->width,
                                           descriptor->normalized_coords,
                                           descriptor->address_mode[0]);
    const int iy = __cumetal_texture_index(y, descriptor->height,
                                           descriptor->normalized_coords,
                                           descriptor->address_mode[1]);
    const int iz = __cumetal_texture_index(z, descriptor->depth,
                                           descriptor->normalized_coords,
                                           descriptor->address_mode[2]);
    return __cumetal_texture_load<T>(descriptor, ix, iy, iz);
}

template <typename T>
__device__ __forceinline__ T texCubemap(cudaTextureObject_t texture,
                                        float x, float y, float z) {
    const float ax = x < 0.0f ? -x : x;
    const float ay = y < 0.0f ? -y : y;
    const float az = z < 0.0f ? -z : z;
    int face = 0;
    float sc = 0.0f;
    float tc = 0.0f;
    float major = 1.0f;
    if (ax >= ay && ax >= az) {
        major = ax;
        if (x >= 0.0f) {
            face = 0;
            sc = -z;
            tc = -y;
        } else {
            face = 1;
            sc = z;
            tc = -y;
        }
    } else if (ay >= az) {
        major = ay;
        if (y >= 0.0f) {
            face = 2;
            sc = x;
            tc = z;
        } else {
            face = 3;
            sc = x;
            tc = -z;
        }
    } else {
        major = az;
        if (z >= 0.0f) {
            face = 4;
            sc = x;
            tc = -y;
        } else {
            face = 5;
            sc = -x;
            tc = -y;
        }
    }
    const float u = 0.5f * (sc / major + 1.0f);
    const float v = 0.5f * (tc / major + 1.0f);
    return tex2DLayered<T>(texture, u, v, face);
}

template <typename T>
__device__ __forceinline__ void surf2Dwrite(T value, cudaSurfaceObject_t surface,
                                            int x_byte, int y,
                                            cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {
    const auto* descriptor = reinterpret_cast<const __cumetal_texture_descriptor*>(surface);
    unsigned char* destination = reinterpret_cast<unsigned char*>(
        descriptor->data + static_cast<unsigned long long>(y) * descriptor->pitch_bytes +
        static_cast<unsigned int>(x_byte));
    *reinterpret_cast<T*>(destination) = value;
}

template <typename KernelFn>
static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, KernelFn* func) {
    return ::cudaFuncGetAttributes(attr, reinterpret_cast<const void*>(func));
}

template <typename KernelFn>
static inline cudaError_t cudaFuncSetCacheConfig(KernelFn* func, cudaFuncCache cacheConfig) {
    return ::cudaFuncSetCacheConfig(reinterpret_cast<const void*>(func), cacheConfig);
}

template <typename KernelFn>
static inline cudaError_t cudaFuncSetSharedMemConfig(KernelFn* func, cudaSharedMemConfig config) {
    return ::cudaFuncSetSharedMemConfig(reinterpret_cast<const void*>(func), config);
}

template <typename KernelFn>
static inline cudaError_t cudaFuncSetAttribute(KernelFn* func, cudaFuncAttribute attr, int value) {
    return ::cudaFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

template <typename KernelFn>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        int* numBlocks, KernelFn* func, int blockSize, size_t dynamicSMemSize = 0) {
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, reinterpret_cast<const void*>(func), blockSize, dynamicSMemSize);
}

template <typename KernelFn>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        int* numBlocks, KernelFn* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, reinterpret_cast<const void*>(func), blockSize, dynamicSMemSize, flags);
}

template <typename KernelFn>
static inline cudaError_t cudaOccupancyMaxPotentialBlockSize(
        int* minGridSize, int* blockSize, KernelFn* func, size_t dynamicSMemSize = 0,
        int blockSizeLimit = 0) {
    return ::cudaOccupancyMaxPotentialBlockSize(
        minGridSize, blockSize, reinterpret_cast<const void*>(func), dynamicSMemSize, blockSizeLimit);
}

static inline cudaError_t cudaMallocManaged(void** dev_ptr, size_t size) {
    return ::cudaMallocManaged(dev_ptr, size, cudaMemAttachGlobal);
}

// Typed cudaMalloc overload — matches the real CUDA SDK signature so that
// code written as `cudaMalloc(&d_ptr, size)` compiles without explicit casts.
template <typename T>
static inline cudaError_t cudaMalloc(T** dev_ptr, size_t size) {
    return ::cudaMalloc(reinterpret_cast<void**>(dev_ptr), size);
}

static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event) {
    return ::cudaStreamWaitEvent(stream, event, 0);
}

// Real CUDA declares these in cuda_runtime.h as typed C++ templates with default
// arguments layered over the C ABI. Without them, ordinary CUDA source that calls
// `cudaMallocHost(&h_ptr, bytes)` or `cudaEventRecord(evt)` does not compile.
template <typename T>
static inline cudaError_t cudaMallocHost(T** ptr, size_t size, unsigned int flags = 0) {
    return flags ? ::cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags)
                 : ::cudaMallocHost(reinterpret_cast<void**>(ptr), size);
}

template <typename T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
    return ::cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags);
}

template <typename T>
static inline cudaError_t cudaHostGetDevicePointer(T** dev_ptr, void* host_ptr,
                                                    unsigned int flags) {
    return ::cudaHostGetDevicePointer(reinterpret_cast<void**>(dev_ptr), host_ptr, flags);
}

// CUDA 13 accepts a typed memory location plus flags and an optional stream.
// CuMetal maps both host and device locations onto the same UMA allocation.
static inline cudaError_t cudaMemPrefetchAsync(const void* dev_ptr, size_t count,
                                                cudaMemLocation location,
                                                unsigned int flags = 0,
                                                cudaStream_t stream = nullptr) {
    if (flags != 0 ||
        (location.type != cudaMemLocationTypeHost &&
         location.type != cudaMemLocationTypeDevice)) {
        return cudaErrorInvalidValue;
    }
    const int destination = location.type == cudaMemLocationTypeHost ? -1 : location.id;
    return ::cudaMemPrefetchAsync(dev_ptr, count, destination, stream);
}

template <typename T>
static inline cudaError_t cudaMallocManaged(T** dev_ptr, size_t size,
                                            unsigned int flags = cudaMemAttachGlobal) {
    return ::cudaMallocManaged(reinterpret_cast<void**>(dev_ptr), size, flags);
}

template <typename T>
static inline cudaError_t cudaMallocAsync(T** dev_ptr, size_t size, cudaStream_t stream) {
    return ::cudaMallocAsync(reinterpret_cast<void**>(dev_ptr), size, stream);
}

static inline cudaError_t cudaEventRecord(cudaEvent_t event) {
    return ::cudaEventRecord(event, 0);
}

static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                          cudaMemcpyKind kind) {
    return ::cudaMemcpyAsync(dst, src, count, kind, 0);
}

static inline cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count) {
    return ::cudaMemsetAsync(dev_ptr, value, count, 0);
}

static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream) {
    return ::cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

// __device__ variables are passed to the symbol APIs by reference, not by address:
// `cudaMemcpyToSymbol(devVar, &host, n)`. The C entry point takes the address.
//
// The reference overloads must NOT swallow pointer arguments. `cudaMemcpyToSymbol(&var, ...)`
// and `cudaMemcpyToSymbol(nullptr, ...)` are the C spelling, and a `const T &`
// template binds them exactly -- beating the `const void *` entry point, which needs
// a pointer conversion -- so the address of the *pointer variable* would be handed to
// the runtime instead of the symbol. Constrain the templates to non-pointer T.
template <typename T>
struct cumetal_symbol_by_ref { static const bool value = true; };
template <typename T>
struct cumetal_symbol_by_ref<T*> { static const bool value = false; };
template <>
struct cumetal_symbol_by_ref<decltype(nullptr)> { static const bool value = false; };

template <bool B, typename T = void>
struct cumetal_symbol_enable_if {};
template <typename T>
struct cumetal_symbol_enable_if<true, T> { typedef T type; };

#define CUMETAL_SYMBOL_BY_REF(T) \
    typename cumetal_symbol_enable_if<cumetal_symbol_by_ref<T>::value, cudaError_t>::type

template <typename T>
static inline CUMETAL_SYMBOL_BY_REF(T) cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count,
                                             size_t offset = 0,
                                             cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    return ::cudaMemcpyToSymbol(reinterpret_cast<const void*>(&symbol), src, count, offset, kind);
}

template <typename T>
static inline CUMETAL_SYMBOL_BY_REF(T) cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count,
                                               size_t offset = 0,
                                               cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
    return ::cudaMemcpyFromSymbol(dst, reinterpret_cast<const void*>(&symbol), count, offset, kind);
}

template <typename T>
static inline CUMETAL_SYMBOL_BY_REF(T) cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count,
                                                  size_t offset, cudaMemcpyKind kind,
                                                  cudaStream_t stream) {
    return ::cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&symbol), src, count, offset,
                                     kind, stream);
}

template <typename T>
static inline CUMETAL_SYMBOL_BY_REF(T) cudaGetSymbolAddress(void** devPtr, const T& symbol) {
    return ::cudaGetSymbolAddress(devPtr, reinterpret_cast<const void*>(&symbol));
}
#endif

#if defined(__cplusplus) && defined(__clang__) && defined(__CUDA__)

#include <limits.h>
#include <math.h>

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>

// Apple's assert() expands to __assert_rtn, which is __host__, so any assert
// inside a __global__/__device__ function is rejected -- in both compilation
// passes, since clang parses device bodies during the host pass too. CUDA solves
// this by shipping a __device__ assert that traps; Metal has no trap-and-report
// path, so the device overload is a no-op and device-side assertions are
// dropped. Declaring a __device__ overload rather than neutering the assert
// macro keeps host-side assertions in .cu files working normally.
#include <cassert>
#if defined(__APPLE__)
__device__ inline void __assert_rtn(const char*, const char*, int, const char*) {}
#else
__device__ inline void __assert_fail(const char*, const char*, unsigned, const char*) {}
#endif

// Optional workaround for CUDA codepaths that reference device-side printf in
// unreachable branches while building with -nocudalib. Use a device stub
// instead of a macro so system headers can still declare host-side printf.
#if defined(CUMETAL_NO_DEVICE_PRINTF)
template <typename... Args>
static __device__ __forceinline__ int printf(const char*, Args...) {
    return 0;
}
#endif

// Device-safe fallback for unqualified `isinf(...)` in CUDA sources when libc++
// only surfaces host overloads in the current include order.
template <typename T>
static __device__ __forceinline__ int isinf(T x) {
    return __builtin_isinf_sign(x) != 0;
}

// NVIDIA's CUDA math overlay provides C++ float overloads in addition to the
// C `*f` spellings. Clang's standalone CUDA overlay only declares
// `double sqrt(double)`, so unqualified `sqrt(float)` would otherwise promote
// to FP64 and emit an unnecessary __nv_sqrt libdevice call.
static __device__ __forceinline__ float sqrt(float x) {
    return __builtin_sqrtf(x);
}

// NVIDIA's C++ math overlay also overloads the unsuffixed spellings for
// binary32. Clang's standalone CUDA overlay exposes only rsqrt(double) and
// fma(double,double,double), plus the C-style rsqrtf/fmaf functions. Without
// these overloads an unqualified FP32 call silently promotes every operand to
// software FP64 and converts the result back, which is both the wrong overload
// contract and catastrophic in inner GPU loops.
static __device__ __forceinline__ float rsqrt(float x) {
    return rsqrtf(x);
}

static __device__ __forceinline__ float fma(float x, float y, float z) {
    return fmaf(x, y, z);
}

static __device__ __forceinline__ float fabs(float x) {
    return __builtin_fabsf(x);
}

// CUDA's C++ math overlay overloads unqualified abs for floating-point
// operands. Clang's standalone CUDA overlay only provides abs(int), which
// otherwise silently converts convergence deltas to integers before taking
// their magnitude.
static __device__ __forceinline__ float abs(float x) {
    return __builtin_fabsf(x);
}

static __device__ __forceinline__ double abs(double x) {
    return __builtin_fabs(x);
}

// Host-only: in device code clang already declares __device__ int min/max, and
// a __host__ __device__ overload of the same signature is rejected outright.
static __host__ __forceinline__ int max(int a, int b) {
    return a > b ? a : b;
}

static __host__ __forceinline__ int min(int a, int b) {
    return a < b ? a : b;
}

// CUDA's math overlay overloads unqualified min/max for floating-point operands,
// the same way it does for abs above. Without these, `min(x, ub)` on two doubles
// picks min(int, int): the PTX comes out as
//     cvt.rzi.s32.f64 -> __nv_min -> cvt.rn.f64.s32
// which truncates both operands to integers and writes the result back as a
// double. That is a silent wrong answer wherever a kernel clamps floating-point
// values, e.g. bound projection in an LP solver.
static __host__ __device__ __forceinline__ float max(float a, float b) {
    return __builtin_fmaxf(a, b);
}

static __host__ __device__ __forceinline__ float min(float a, float b) {
    return __builtin_fminf(a, b);
}

static __host__ __device__ __forceinline__ double max(double a, double b) {
    return __builtin_fmax(a, b);
}

static __host__ __device__ __forceinline__ double min(double a, double b) {
    return __builtin_fmin(a, b);
}

// CUDA promotes mixed float/double arguments to double rather than letting the
// call become ambiguous.
static __host__ __device__ __forceinline__ double max(float a, double b) {
    return __builtin_fmax((double) a, b);
}

static __host__ __device__ __forceinline__ double max(double a, float b) {
    return __builtin_fmax(a, (double) b);
}

static __host__ __device__ __forceinline__ double min(float a, double b) {
    return __builtin_fmin((double) a, b);
}

static __host__ __device__ __forceinline__ double min(double a, float b) {
    return __builtin_fmin(a, (double) b);
}

template <typename T>
static __device__ __forceinline__ T __ldcs(const T* ptr) {
    return *ptr;
}
template <typename T>
static __device__ __forceinline__ T __ldca(const T* ptr) {
    return *ptr;
}
template <typename T>
static __device__ __forceinline__ T __ldcg(const T* ptr) {
    return *ptr;
}
template <typename T>
static __device__ __forceinline__ T __ldlu(const T* ptr) {
    return *ptr;
}
template <typename T>
static __device__ __forceinline__ T __ldcv(const T* ptr) {
    return *ptr;
}
template <typename T>
static __device__ __forceinline__ void __stwb(T* ptr, T value) {
    *ptr = value;
}
template <typename T>
static __device__ __forceinline__ void __stcg(T* ptr, T value) {
    *ptr = value;
}
template <typename T>
static __device__ __forceinline__ void __stcs(T* ptr, T value) {
    *ptr = value;
}
template <typename T>
static __device__ __forceinline__ void __stwt(T* ptr, T value) {
    *ptr = value;
}

static __device__ __forceinline__ unsigned int __cvta_generic_to_shared(const void* generic_ptr) {
    unsigned int shared_ptr;
    asm("cvta.to.shared.u32 %0, %1;" : "=r"(shared_ptr) : "l"(generic_ptr));
    return shared_ptr;
}

// __ldg: load via read-only (texture) cache. On UMA Apple Silicon there is no
// dedicated read-only cache, so this is a plain load — identical semantics,
// no performance difference (spec §8).
template <typename T>
static __device__ __forceinline__ T __ldg(const T* ptr) {
    return *ptr;
}

// ── Atomic operations ────────────────────────────────────────────────────────

static __device__ __forceinline__ float atomicAdd(float* ptr, float val) {
    // Clang lowers its generic NVVM floating atomic to a system-scope CAS loop
    // for this standalone CUDA overlay. That is correct but catastrophic for
    // force accumulation: a native CUDA float atomic is directly representable
    // in PTX and AIR. Spell it explicitly so the PTX-to-AIR path can select
    // Metal's native atomic_float add instead of executing the retry loop.
    float old;
    asm volatile("atom.global.add.f32 %0, [%1], %2;"
                 : "=f"(old) : "l"(ptr), "f"(val) : "memory");
    return old;
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

// The rest of CUDA's 64-bit atomic surface. These were missing outright, so a
// program that used them failed to compile rather than failing to lower -- and
// they are ordinary CUDA, not an exotic corner. Metal still has no atomic this
// wide; each one arrives at the PTX lowering as an atom.*.b64/u64 and is
// serialized behind the address-hashed lock bank.
static __device__ __forceinline__ unsigned long long atomicExch(unsigned long long* ptr,
                                                                 unsigned long long val) {
    return __ullAtomicExch(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicMin(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicMin(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicMax(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicMax(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicAnd(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicAnd(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicOr(unsigned long long* ptr,
                                                               unsigned long long val) {
    return __ullAtomicOr(ptr, val);
}
static __device__ __forceinline__ unsigned long long atomicXor(unsigned long long* ptr,
                                                                unsigned long long val) {
    return __ullAtomicXor(ptr, val);
}
static __device__ __forceinline__ long long atomicAnd(long long* ptr, long long val) {
    return __llAtomicAnd(ptr, val);
}
static __device__ __forceinline__ long long atomicOr(long long* ptr, long long val) {
    return __llAtomicOr(ptr, val);
}
static __device__ __forceinline__ long long atomicXor(long long* ptr, long long val) {
    return __llAtomicXor(ptr, val);
}
// Clang exposes no signed 64-bit min/max builtin, so these take the CAS loop --
// the same construction CUDA itself uses below sm_35, and the same one clang
// gives atomicAdd(double*).
static __device__ __forceinline__ long long atomicMin(long long* ptr, long long val) {
    unsigned long long* base = reinterpret_cast<unsigned long long*>(ptr);
    unsigned long long old = *base;
    unsigned long long assumed;
    do {
        assumed = old;
        const long long cur = static_cast<long long>(assumed);
        if (cur <= val) break;
        old = __ullAtomicCAS(base, assumed, static_cast<unsigned long long>(val));
    } while (old != assumed);
    return static_cast<long long>(old);
}
static __device__ __forceinline__ long long atomicMax(long long* ptr, long long val) {
    unsigned long long* base = reinterpret_cast<unsigned long long*>(ptr);
    unsigned long long old = *base;
    unsigned long long assumed;
    do {
        assumed = old;
        const long long cur = static_cast<long long>(assumed);
        if (cur >= val) break;
        old = __ullAtomicCAS(base, assumed, static_cast<unsigned long long>(val));
    } while (old != assumed);
    return static_cast<long long>(old);
}

// atomicInc/atomicDec are the wrapping counters CUDA exposes only for unsigned.
// There is no single Metal instruction for them, so build them on a CAS loop --
// same construction the double-precision atomicAdd above uses.
static __device__ __forceinline__ unsigned int atomicInc(unsigned int* ptr, unsigned int val) {
    unsigned int old = *ptr;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int next = (assumed >= val) ? 0u : (assumed + 1u);
        old = __uAtomicCAS(ptr, assumed, next);
    } while (assumed != old);
    return old;
}

static __device__ __forceinline__ unsigned int atomicDec(unsigned int* ptr, unsigned int val) {
    unsigned int old = *ptr;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned int next = (assumed == 0u || assumed > val) ? val : (assumed - 1u);
        old = __uAtomicCAS(ptr, assumed, next);
    } while (assumed != old);
    return old;
}

// System-scope variants use Clang's documented CUDA NVVM wrappers. On Apple
// Silicon, managed allocations are shared UMA buffers, so these reach the same
// bytes as host atomics. Keep the `_system` spelling distinct: collapsing it to
// ordinary device-scope helpers would compile but would not express host/device
// atomic visibility in the generated PTX.
static __device__ __forceinline__ int atomicAdd_system(int* ptr, int val) {
    return __iAtomicAdd_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicAdd_system(unsigned int* ptr,
                                                                 unsigned int val) {
    return __uAtomicAdd_system(ptr, val);
}
static __device__ __forceinline__ int atomicExch_system(int* ptr, int val) {
    return __iAtomicExch_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicExch_system(unsigned int* ptr,
                                                                  unsigned int val) {
    return __uAtomicExch_system(ptr, val);
}
static __device__ __forceinline__ int atomicMin_system(int* ptr, int val) {
    return __iAtomicMin_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicMin_system(unsigned int* ptr,
                                                                 unsigned int val) {
    return __uAtomicMin_system(ptr, val);
}
static __device__ __forceinline__ int atomicMax_system(int* ptr, int val) {
    return __iAtomicMax_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicMax_system(unsigned int* ptr,
                                                                 unsigned int val) {
    return __uAtomicMax_system(ptr, val);
}
static __device__ __forceinline__ int atomicCAS_system(int* ptr, int cmp, int val) {
    return __iAtomicCAS_system(ptr, cmp, val);
}
static __device__ __forceinline__ unsigned int atomicCAS_system(unsigned int* ptr,
                                                                 unsigned int cmp,
                                                                 unsigned int val) {
    return __uAtomicCAS_system(ptr, cmp, val);
}
static __device__ __forceinline__ int atomicAnd_system(int* ptr, int val) {
    return __iAtomicAnd_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicAnd_system(unsigned int* ptr,
                                                                 unsigned int val) {
    return __uAtomicAnd_system(ptr, val);
}
static __device__ __forceinline__ int atomicOr_system(int* ptr, int val) {
    return __iAtomicOr_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicOr_system(unsigned int* ptr,
                                                                unsigned int val) {
    return __uAtomicOr_system(ptr, val);
}
static __device__ __forceinline__ int atomicXor_system(int* ptr, int val) {
    return __iAtomicXor_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicXor_system(unsigned int* ptr,
                                                                 unsigned int val) {
    return __uAtomicXor_system(ptr, val);
}
static __device__ __forceinline__ unsigned int atomicInc_system(unsigned int* ptr,
                                                                 unsigned int val) {
    unsigned int old = *ptr;
    unsigned int assumed;
    do {
        assumed = old;
        const unsigned int next = (assumed >= val) ? 0u : (assumed + 1u);
        old = __uAtomicCAS_system(ptr, assumed, next);
    } while (assumed != old);
    return old;
}
static __device__ __forceinline__ unsigned int atomicDec_system(unsigned int* ptr,
                                                                 unsigned int val) {
    unsigned int old = *ptr;
    unsigned int assumed;
    do {
        assumed = old;
        const unsigned int next = (assumed == 0u || assumed > val) ? val : (assumed - 1u);
        old = __uAtomicCAS_system(ptr, assumed, next);
    } while (assumed != old);
    return old;
}

// ── Synchronization, memory fences, bit ops, FMA ────────────────────────────
// Sync shuffle/vote/reduce wrappers live in clang's __clang_cuda_intrinsics.h.
// When building with -nocudainc we don't include that header, so provide them.
#ifndef __CLANG_CUDA_INTRINSICS_H__

static __device__ __forceinline__ void __syncwarp(unsigned int mask = 0xffffffffu) {
    __nvvm_bar_warp_sync(mask);
}

static __device__ __forceinline__ int __cumetal_shfl_clamp(int width, int lane_mask) {
    return ((32 - width) << 8) | lane_mask;
}

static __device__ __forceinline__ int __cumetal_shfl_sync_idx_i32(unsigned int mask, int val, int srcLane, int clamp) {
    int out;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
                 : "=r"(out)
                 : "r"(val), "r"(srcLane), "r"(clamp), "r"(mask));
    return out;
}

static __device__ __forceinline__ int __cumetal_shfl_sync_down_i32(unsigned int mask, int val, unsigned int delta, int clamp) {
    int out;
    asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
                 : "=r"(out)
                 : "r"(val), "r"(delta), "r"(clamp), "r"(mask));
    return out;
}

static __device__ __forceinline__ int __cumetal_shfl_sync_up_i32(unsigned int mask, int val, unsigned int delta, int clamp) {
    int out;
    asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
                 : "=r"(out)
                 : "r"(val), "r"(delta), "r"(clamp), "r"(mask));
    return out;
}

static __device__ __forceinline__ float __cumetal_shfl_i32_bits_to_f32(int bits) {
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

static __device__ __forceinline__ int __cumetal_shfl_f32_bits_to_i32(float val) {
    int bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    return bits;
}

static __device__ __forceinline__ int __cumetal_shfl_u32_bits_to_i32(unsigned int val) {
    int bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    return bits;
}

static __device__ __forceinline__ unsigned int __cumetal_shfl_i32_bits_to_u32(int bits) {
    unsigned int out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

// Warp shuffle intrinsics (spec §5.3).
// On Apple Silicon the full warp participates; partial masks are conservative no-ops.
static __device__ __forceinline__ int __shfl_sync(unsigned int mask, int val, int srcLane, int width = 32) {
    return __cumetal_shfl_sync_idx_i32(mask, val, srcLane, __cumetal_shfl_clamp(width, 0x1f));
}
static __device__ __forceinline__ float __shfl_sync(unsigned int mask, float val, int srcLane, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_idx_i32(mask, __cumetal_shfl_f32_bits_to_i32(val), srcLane,
                                                     __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_f32(out_bits);
}
static __device__ __forceinline__ unsigned int __shfl_sync(unsigned int mask, unsigned int val, int srcLane, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_idx_i32(mask, __cumetal_shfl_u32_bits_to_i32(val), srcLane,
                                                     __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_u32(out_bits);
}
static __device__ __forceinline__ unsigned long long __shfl_sync(
    unsigned int mask, unsigned long long val, int srcLane, int width = 32) {
    const unsigned int lo = static_cast<unsigned int>(val);
    const unsigned int hi = static_cast<unsigned int>(val >> 32);
    return static_cast<unsigned long long>(__shfl_sync(mask, lo, srcLane, width)) |
           (static_cast<unsigned long long>(__shfl_sync(mask, hi, srcLane, width)) << 32);
}
static __device__ __forceinline__ long long __shfl_sync(
    unsigned int mask, long long val, int srcLane, int width = 32) {
    return static_cast<long long>(
        __shfl_sync(mask, static_cast<unsigned long long>(val), srcLane, width));
}
static __device__ __forceinline__ double __shfl_sync(
    unsigned int mask, double val, int srcLane, int width = 32) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    bits = __shfl_sync(mask, bits, srcLane, width);
    double out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}
static __device__ __forceinline__ int __shfl_down_sync(unsigned int mask, int val, unsigned int delta, int width = 32) {
    return __cumetal_shfl_sync_down_i32(mask, val, delta, __cumetal_shfl_clamp(width, 0x1f));
}
static __device__ __forceinline__ float __shfl_down_sync(unsigned int mask, float val, unsigned int delta, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_down_i32(mask, __cumetal_shfl_f32_bits_to_i32(val), delta,
                                                      __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_f32(out_bits);
}
static __device__ __forceinline__ unsigned int __shfl_down_sync(unsigned int mask, unsigned int val, unsigned int delta, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_down_i32(mask, __cumetal_shfl_u32_bits_to_i32(val), delta,
                                                      __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_u32(out_bits);
}
static __device__ __forceinline__ unsigned long long __shfl_down_sync(
    unsigned int mask, unsigned long long val, unsigned int delta, int width = 32) {
    const unsigned int lo = static_cast<unsigned int>(val);
    const unsigned int hi = static_cast<unsigned int>(val >> 32);
    return static_cast<unsigned long long>(__shfl_down_sync(mask, lo, delta, width)) |
           (static_cast<unsigned long long>(__shfl_down_sync(mask, hi, delta, width)) << 32);
}
static __device__ __forceinline__ long long __shfl_down_sync(
    unsigned int mask, long long val, unsigned int delta, int width = 32) {
    return static_cast<long long>(
        __shfl_down_sync(mask, static_cast<unsigned long long>(val), delta, width));
}
static __device__ __forceinline__ double __shfl_down_sync(
    unsigned int mask, double val, unsigned int delta, int width = 32) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    bits = __shfl_down_sync(mask, bits, delta, width);
    double out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}
static __device__ __forceinline__ int __shfl_up_sync(unsigned int mask, int val, unsigned int delta, int width = 32) {
    return __cumetal_shfl_sync_up_i32(mask, val, delta, __cumetal_shfl_clamp(width, 0));
}
static __device__ __forceinline__ float __shfl_up_sync(unsigned int mask, float val, unsigned int delta, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_up_i32(mask, __cumetal_shfl_f32_bits_to_i32(val), delta,
                                                    __cumetal_shfl_clamp(width, 0));
    return __cumetal_shfl_i32_bits_to_f32(out_bits);
}
static __device__ __forceinline__ unsigned int __shfl_up_sync(unsigned int mask, unsigned int val, unsigned int delta, int width = 32) {
    const int out_bits = __cumetal_shfl_sync_up_i32(mask, __cumetal_shfl_u32_bits_to_i32(val), delta,
                                                    __cumetal_shfl_clamp(width, 0));
    return __cumetal_shfl_i32_bits_to_u32(out_bits);
}
static __device__ __forceinline__ unsigned long long __shfl_up_sync(
    unsigned int mask, unsigned long long val, unsigned int delta, int width = 32) {
    const unsigned int lo = static_cast<unsigned int>(val);
    const unsigned int hi = static_cast<unsigned int>(val >> 32);
    return static_cast<unsigned long long>(__shfl_up_sync(mask, lo, delta, width)) |
           (static_cast<unsigned long long>(__shfl_up_sync(mask, hi, delta, width)) << 32);
}
static __device__ __forceinline__ long long __shfl_up_sync(
    unsigned int mask, long long val, unsigned int delta, int width = 32) {
    return static_cast<long long>(
        __shfl_up_sync(mask, static_cast<unsigned long long>(val), delta, width));
}
static __device__ __forceinline__ double __shfl_up_sync(
    unsigned int mask, double val, unsigned int delta, int width = 32) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    bits = __shfl_up_sync(mask, bits, delta, width);
    double out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}
static __device__ __forceinline__ int __shfl_xor_sync(unsigned int mask, int val, int laneMask, int width = 32) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    const int srcLane = static_cast<int>(laneid) ^ laneMask;
    return __cumetal_shfl_sync_idx_i32(mask, val, srcLane, __cumetal_shfl_clamp(width, 0x1f));
}
static __device__ __forceinline__ float __shfl_xor_sync(unsigned int mask, float val, int laneMask, int width = 32) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    const int srcLane = static_cast<int>(laneid) ^ laneMask;
    const int out_bits = __cumetal_shfl_sync_idx_i32(mask, __cumetal_shfl_f32_bits_to_i32(val), srcLane,
                                                     __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_f32(out_bits);
}
static __device__ __forceinline__ unsigned int __shfl_xor_sync(unsigned int mask, unsigned int val, int laneMask, int width = 32) {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    const int srcLane = static_cast<int>(laneid) ^ laneMask;
    const int out_bits = __cumetal_shfl_sync_idx_i32(mask, __cumetal_shfl_u32_bits_to_i32(val), srcLane,
                                                     __cumetal_shfl_clamp(width, 0x1f));
    return __cumetal_shfl_i32_bits_to_u32(out_bits);
}
static __device__ __forceinline__ unsigned long long __shfl_xor_sync(
    unsigned int mask, unsigned long long val, int laneMask, int width = 32) {
    const unsigned int lo = static_cast<unsigned int>(val);
    const unsigned int hi = static_cast<unsigned int>(val >> 32);
    return static_cast<unsigned long long>(__shfl_xor_sync(mask, lo, laneMask, width)) |
           (static_cast<unsigned long long>(__shfl_xor_sync(mask, hi, laneMask, width)) << 32);
}
static __device__ __forceinline__ long long __shfl_xor_sync(
    unsigned int mask, long long val, int laneMask, int width = 32) {
    return static_cast<long long>(
        __shfl_xor_sync(mask, static_cast<unsigned long long>(val), laneMask, width));
}
static __device__ __forceinline__ double __shfl_xor_sync(
    unsigned int mask, double val, int laneMask, int width = 32) {
    unsigned long long bits;
    __builtin_memcpy(&bits, &val, sizeof(bits));
    bits = __shfl_xor_sync(mask, bits, laneMask, width);
    double out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

// Warp vote intrinsics (spec §5.3).
static __device__ __forceinline__ int __any_sync(unsigned int mask, int predicate) {
    return __nvvm_vote_any_sync(mask, predicate);
}
static __device__ __forceinline__ int __all_sync(unsigned int mask, int predicate) {
    return __nvvm_vote_all_sync(mask, predicate);
}
static __device__ __forceinline__ unsigned int __ballot_sync(unsigned int mask, int predicate) {
    return __nvvm_vote_ballot_sync(mask, predicate);
}

static __device__ __forceinline__ unsigned int __activemask(void) {
    unsigned int active;
    asm("mov.u32 %0, %%activemask;" : "=r"(active));
    return active;
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
// Implement via warp shuffles for broad clang/PTX compatibility.
static __device__ __forceinline__ unsigned int __reduce_add_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val;
}
static __device__ __forceinline__ int __reduce_add_sync(unsigned int mask, int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val;
}
static __device__ __forceinline__ unsigned int __reduce_and_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val &= __shfl_xor_sync(mask, val, offset);
    }
    return val;
}
static __device__ __forceinline__ unsigned int __reduce_or_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val |= __shfl_xor_sync(mask, val, offset);
    }
    return val;
}
static __device__ __forceinline__ unsigned int __reduce_xor_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val ^= __shfl_xor_sync(mask, val, offset);
    }
    return val;
}
static __device__ __forceinline__ unsigned int __reduce_min_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned int other = __shfl_xor_sync(mask, val, offset);
        val = other < val ? other : val;
    }
    return val;
}
static __device__ __forceinline__ int __reduce_min_sync(unsigned int mask, int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(mask, val, offset);
        val = other < val ? other : val;
    }
    return val;
}
static __device__ __forceinline__ unsigned int __reduce_max_sync(unsigned int mask, unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned int other = __shfl_xor_sync(mask, val, offset);
        val = other > val ? other : val;
    }
    return val;
}
static __device__ __forceinline__ int __reduce_max_sync(unsigned int mask, int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(mask, val, offset);
        val = other > val ? other : val;
    }
    return val;
}

#endif  // !__CLANG_CUDA_INTRINSICS_H__

// Only define when clang's __clang_cuda_device_functions.h hasn't already
// provided these (it uses the guard __CLANG_CUDA_DEVICE_FUNCTIONS_H__).
#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__

static __device__ __forceinline__ void __threadfence(void) { __nvvm_membar_gl(); }
static __device__ __forceinline__ void __threadfence_block(void) { __nvvm_membar_cta(); }
static __device__ __forceinline__ void __threadfence_system(void) { __nvvm_membar_sys(); }

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

// CUDA's math headers make nan() callable from device code, but Clang's
// -nocudainc wrapper leaves libc's declaration host-only. Keep this outside
// __CLANG_CUDA_DEVICE_FUNCTIONS_H__: recent Clang defines that guard without
// supplying nan().
#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ double __cumetal_nan(const char*) {
    return __builtin_nan("");
}
#define nan __cumetal_nan
#endif

// Integer dot-product intrinsic (4x int8 -> int32 accumulate). Clang's CUDA
// headers may not provide __dp4a in CUDA mode without NVIDIA headers.
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

#endif  // device code section
