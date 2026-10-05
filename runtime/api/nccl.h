#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef NCCL_H_
#define NCCL_H_ 1
#endif


#include "cuda_runtime.h"

#include <limits.h>
#include <stddef.h>

// Must agree with ncclGetVersion. Callers pick API generations from these at
// compile time, and an absent NCCL_MAJOR reads as NCCL 1.x.
#define NCCL_MAJOR 2
#define NCCL_MINOR 18
#define NCCL_PATCH 0
#define NCCL_SUFFIX ""
#define NCCL_VERSION(X, Y, Z) \
    (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))
#define NCCL_VERSION_CODE NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ncclComm* ncclComm_t;

typedef enum ncclResult_t {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
} ncclResult_t;

typedef enum ncclDataType_t {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
} ncclDataType_t;

typedef enum ncclRedOp_t {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
} ncclRedOp_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

#define NCCL_SPLIT_NOCOLOR -1
#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL

typedef struct ncclConfig_v21700 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    int blocking;
    int cgaClusterSize;
    int minCTAs;
    int maxCTAs;
    const char* netName;
    int splitShare;
} ncclConfig_t;

#define NCCL_CONFIG_INITIALIZER                                                       \
    {                                                                                 \
        sizeof(ncclConfig_t), 0xcafebeef, NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), \
            NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT,      \
            NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_PTR, NCCL_CONFIG_UNDEF_INT        \
    }

// Version
ncclResult_t ncclGetVersion(int* version);
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);

// Communicator management
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                    int rank, ncclConfig_t* config);
// A single-rank communicator splits into itself (or into nothing for
// NCCL_SPLIT_NOCOLOR).
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm,
                           ncclConfig_t* config);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

// Collective operations (single-GPU: operate in-place or memcpy)
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, cudaStream_t stream);
// Legacy in-place broadcast; with one rank the buffer already holds root's data.
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
                       ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                       int peer, ncclComm_t comm, cudaStream_t stream);

// Group operations
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

#ifdef __cplusplus
}
#endif
