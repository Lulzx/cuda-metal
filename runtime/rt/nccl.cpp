#include "nccl.h"

#include <cstdlib>
#include <cstring>
#include <new>

// NCCL shim for single-GPU Apple Silicon.
// All collective ops are identity (single rank) — just memcpy sendbuff to recvbuff.

struct ncclComm {
    int nranks = 1;
    int rank = 0;
    int device = 0;
};

namespace {

size_t nccl_dtype_size(ncclDataType_t dt) {
    switch (dt) {
        case ncclInt8:
        case ncclUint8: return 1;
        case ncclFloat16:
        case ncclBfloat16: return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32: return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64: return 8;
        default: return 4;
    }
}

} // namespace

extern "C" {

ncclResult_t ncclGetVersion(int* version) {
    if (!version) return ncclInvalidArgument;
    *version = 21800; // 2.18.0
    return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "no error";
        case ncclUnhandledCudaError: return "unhandled cuda error";
        case ncclSystemError: return "unhandled system error";
        case ncclInternalError: return "internal error";
        case ncclInvalidArgument: return "invalid argument";
        case ncclInvalidUsage: return "invalid usage";
        case ncclRemoteError: return "remote process exited or there was a network error";
        case ncclInProgress: return "NCCL operation in progress";
        default: return "unknown error";
    }
}

const char* ncclGetLastError(ncclComm_t /*comm*/) {
    return nullptr; // no errors tracked
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    *uniqueId = 1;
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId /*commId*/, int rank) {
    if (!comm) return ncclInvalidArgument;
    if (nranks != 1 || rank != 0) return ncclInvalidArgument; // single GPU only
    auto* c = new (std::nothrow) ncclComm;
    if (!c) return ncclInternalError;
    c->nranks = 1;
    c->rank = 0;
    c->device = 0;
    *comm = c;
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* /*devlist*/) {
    if (!comms || ndev < 1) return ncclInvalidArgument;
    return ncclCommInitRank(&comms[0], 1, 1, 0);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    delete comm;
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    delete comm;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!comm || !count) return ncclInvalidArgument;
    *count = comm->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
    if (!comm || !device) return ncclInvalidArgument;
    *device = comm->device;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    *rank = comm->rank;
    return ncclSuccess;
}

// Single-rank collectives: just copy sendbuff -> recvbuff

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t /*op*/,
                            ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    if (!sendbuff || !recvbuff) return ncclInvalidArgument;
    if (sendbuff != recvbuff)
        std::memcpy(recvbuff, sendbuff, count * nccl_dtype_size(datatype));
    return ncclSuccess;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int /*root*/,
                            ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    if (!sendbuff || !recvbuff) return ncclInvalidArgument;
    if (sendbuff != recvbuff)
        std::memcpy(recvbuff, sendbuff, count * nccl_dtype_size(datatype));
    return ncclSuccess;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t /*op*/, int /*root*/,
                         ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    if (!sendbuff || !recvbuff) return ncclInvalidArgument;
    if (sendbuff != recvbuff)
        std::memcpy(recvbuff, sendbuff, count * nccl_dtype_size(datatype));
    return ncclSuccess;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype,
                            ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    if (!sendbuff || !recvbuff) return ncclInvalidArgument;
    if (sendbuff != recvbuff)
        std::memcpy(recvbuff, sendbuff, sendcount * nccl_dtype_size(datatype));
    return ncclSuccess;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                ncclDataType_t datatype, ncclRedOp_t /*op*/,
                                ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    if (!sendbuff || !recvbuff) return ncclInvalidArgument;
    if (sendbuff != recvbuff)
        std::memcpy(recvbuff, sendbuff, recvcount * nccl_dtype_size(datatype));
    return ncclSuccess;
}

ncclResult_t ncclSend(const void* /*sendbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/,
                       int /*peer*/, ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    return ncclInvalidUsage; // No peers on single GPU
}

ncclResult_t ncclRecv(void* /*recvbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/,
                       int /*peer*/, ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    return ncclInvalidUsage; // No peers on single GPU
}

ncclResult_t ncclGroupStart(void) { return ncclSuccess; }
ncclResult_t ncclGroupEnd(void) { return ncclSuccess; }

} // extern "C"
