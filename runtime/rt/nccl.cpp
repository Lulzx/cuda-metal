#include "nccl.h"

#include <cstdlib>
#include <limits>
#include <new>

// NCCL shim for single-GPU Apple Silicon.
// All collective ops are identity (single rank) — just memcpy sendbuff to recvbuff.

struct ncclComm {
    int nranks = 1;
    int rank = 0;
    int device = 0;
};

namespace {

bool nccl_dtype_size(ncclDataType_t dt, size_t* size) {
    if (!size) return false;
    switch (dt) {
        case ncclInt8:
        case ncclUint8: *size = 1; return true;
        case ncclFloat16:
        case ncclBfloat16: *size = 2; return true;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32: *size = 4; return true;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64: *size = 8; return true;
        default: return false;
    }
}

bool valid_reduction(ncclRedOp_t op) {
    return op == ncclSum || op == ncclProd || op == ncclMax ||
           op == ncclMin || op == ncclAvg;
}

bool valid_comm(ncclComm_t comm) {
    return comm && comm->nranks == 1 && comm->rank == 0 && comm->device == 0;
}

ncclResult_t collective_bytes(ncclComm_t comm, size_t count,
                              ncclDataType_t datatype, size_t* bytes) {
    if (!valid_comm(comm) || !bytes) return ncclInvalidArgument;
    size_t element_size = 0;
    if (!nccl_dtype_size(datatype, &element_size)) return ncclInvalidArgument;
    if (count > std::numeric_limits<size_t>::max() / element_size)
        return ncclInvalidArgument;
    *bytes = count * element_size;
    return ncclSuccess;
}

thread_local bool group_active = false;

ncclResult_t enqueue_identity_copy(const void* sendbuff, void* recvbuff,
                                   size_t bytes, cudaStream_t stream) {
    if (bytes == 0 || sendbuff == recvbuff) return ncclSuccess;
    if (sendbuff == nullptr || recvbuff == nullptr) return ncclInvalidArgument;
    return cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDefault, stream) == cudaSuccess
               ? ncclSuccess
               : ncclUnhandledCudaError;
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

const char* ncclGetLastError(ncclComm_t comm) {
    return valid_comm(comm) ? "no error" : "invalid communicator";
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    *uniqueId = 1;
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId /*commId*/, int rank) {
    if (!comm) return ncclInvalidArgument;
    *comm = nullptr;
    if (nranks != 1 || rank != 0) return ncclInvalidArgument; // single GPU only
    auto* c = new (std::nothrow) ncclComm;
    if (!c) return ncclInternalError;
    c->nranks = 1;
    c->rank = 0;
    c->device = 0;
    *comm = c;
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
    if (!comms || ndev != 1 || (devlist && devlist[0] != 0))
        return ncclInvalidArgument;
    return ncclCommInitRank(&comms[0], 1, 1, 0);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!valid_comm(comm)) return ncclInvalidArgument;
    delete comm;
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    if (!valid_comm(comm)) return ncclInvalidArgument;
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
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream) {
    if (!valid_reduction(op)) return ncclInvalidArgument;
    size_t bytes = 0;
    ncclResult_t status = collective_bytes(comm, count, datatype, &bytes);
    return status == ncclSuccess ? enqueue_identity_copy(sendbuff, recvbuff, bytes, stream) : status;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, cudaStream_t stream) {
    if (root != 0) return ncclInvalidArgument;
    size_t bytes = 0;
    ncclResult_t status = collective_bytes(comm, count, datatype, &bytes);
    return status == ncclSuccess ? enqueue_identity_copy(sendbuff, recvbuff, bytes, stream) : status;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream) {
    if (!valid_reduction(op) || root != 0) return ncclInvalidArgument;
    size_t bytes = 0;
    ncclResult_t status = collective_bytes(comm, count, datatype, &bytes);
    return status == ncclSuccess ? enqueue_identity_copy(sendbuff, recvbuff, bytes, stream) : status;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype,
                            ncclComm_t comm, cudaStream_t stream) {
    size_t bytes = 0;
    ncclResult_t status = collective_bytes(comm, sendcount, datatype, &bytes);
    return status == ncclSuccess ? enqueue_identity_copy(sendbuff, recvbuff, bytes, stream) : status;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                ncclComm_t comm, cudaStream_t stream) {
    if (!valid_reduction(op)) return ncclInvalidArgument;
    size_t bytes = 0;
    ncclResult_t status = collective_bytes(comm, recvcount, datatype, &bytes);
    return status == ncclSuccess ? enqueue_identity_copy(sendbuff, recvbuff, bytes, stream) : status;
}

ncclResult_t ncclSend(const void* /*sendbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/,
                       int /*peer*/, ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    return ncclInvalidUsage; // No peers on single GPU
}

ncclResult_t ncclRecv(void* /*recvbuff*/, size_t /*count*/, ncclDataType_t /*datatype*/,
                       int /*peer*/, ncclComm_t /*comm*/, cudaStream_t /*stream*/) {
    return ncclInvalidUsage; // No peers on single GPU
}

ncclResult_t ncclGroupStart(void) {
    if (group_active) return ncclInvalidUsage;
    group_active = true;
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    if (!group_active) return ncclInvalidUsage;
    group_active = false;
    return ncclSuccess;
}

} // extern "C"
