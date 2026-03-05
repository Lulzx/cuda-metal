#pragma once

#include "cuda_runtime.h"

#include <stddef.h>

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

typedef int ncclUniqueId;

// Version
ncclResult_t ncclGetVersion(int* version);
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);

// Communicator management
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist);
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
