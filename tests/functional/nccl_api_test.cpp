#include "nccl.h"

#include <cstdio>
#include <cstring>
#include <limits>

static bool test_version() {
    int ver = 0;
    ncclResult_t r = ncclGetVersion(&ver);
    if (r != ncclSuccess || ver == 0) {
        std::fprintf(stderr, "FAIL: ncclGetVersion returned %d, ver=%d\n", r, ver);
        return false;
    }
    return true;
}

static bool test_comm_lifecycle() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    ncclComm_t comm = nullptr;
    ncclResult_t r = ncclCommInitRank(&comm, 1, id, 0);
    if (r != ncclSuccess || comm == nullptr) {
        std::fprintf(stderr, "FAIL: ncclCommInitRank returned %d\n", r);
        return false;
    }

    int count = 0, device = -1, rank = -1;
    ncclCommCount(comm, &count);
    ncclCommCuDevice(comm, &device);
    ncclCommUserRank(comm, &rank);

    if (count != 1 || device != 0 || rank != 0) {
        std::fprintf(stderr, "FAIL: comm count=%d device=%d rank=%d\n", count, device, rank);
        return false;
    }

    r = ncclCommDestroy(comm);
    if (r != ncclSuccess) {
        std::fprintf(stderr, "FAIL: ncclCommDestroy returned %d\n", r);
        return false;
    }
    return true;
}

static bool test_allreduce_identity() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm = nullptr;
    ncclCommInitRank(&comm, 1, id, 0);

    float send[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float recv[4] = {};

    ncclResult_t r = ncclAllReduce(send, recv, 4, ncclFloat32, ncclSum, comm, nullptr);
    if (r != ncclSuccess) {
        std::fprintf(stderr, "FAIL: ncclAllReduce returned %d\n", r);
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    for (int i = 0; i < 4; ++i) {
        if (recv[i] != send[i]) {
            std::fprintf(stderr, "FAIL: allreduce[%d]=%f expected %f\n", i, recv[i], send[i]);
            return false;
        }
    }

    ncclCommDestroy(comm);
    return true;
}

static bool test_broadcast_identity() {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm = nullptr;
    ncclCommInitRank(&comm, 1, id, 0);

    float data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out[4] = {};
    ncclBroadcast(data, out, 4, ncclFloat32, 0, comm, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    for (int i = 0; i < 4; ++i) {
        if (out[i] != data[i]) {
            std::fprintf(stderr, "FAIL: broadcast[%d]=%f expected %f\n", i, out[i], data[i]);
            return false;
        }
    }

    ncclCommDestroy(comm);
    return true;
}

static bool test_error_string() {
    const char* s = ncclGetErrorString(ncclSuccess);
    if (!s || std::strlen(s) == 0) return false;
    s = ncclGetErrorString(ncclInvalidArgument);
    if (!s || std::strlen(s) == 0) return false;
    return true;
}

static bool test_multi_rank_rejected() {
    ncclComm_t comm = nullptr;
    ncclResult_t r = ncclCommInitRank(&comm, 2, 1, 0);
    if (r == ncclSuccess) {
        std::fprintf(stderr, "FAIL: multi-rank should be rejected\n");
        return false;
    }
    return true;
}

static bool test_single_rank_validation() {
    ncclComm_t comm = nullptr;
    ncclUniqueId id = 0;
    ncclGetUniqueId(&id);
    if (ncclCommInitRank(&comm, 1, id, 0) != ncclSuccess) return false;

    float send = 3.0f, recv = -7.0f;
    if (ncclAllReduce(&send, &recv, 1, static_cast<ncclDataType_t>(999),
                      ncclSum, comm, nullptr) != ncclInvalidArgument || recv != -7.0f ||
        ncclAllReduce(&send, &recv, 1, ncclFloat32,
                      static_cast<ncclRedOp_t>(999), comm, nullptr) != ncclInvalidArgument ||
        ncclBroadcast(&send, &recv, 1, ncclFloat32, 1, comm, nullptr) != ncclInvalidArgument ||
        ncclAllReduce(&send, &recv, 1, ncclFloat32, ncclSum,
                      nullptr, nullptr) != ncclInvalidArgument ||
        ncclAllReduce(&send, &recv, std::numeric_limits<size_t>::max(),
                      ncclFloat64, ncclSum, comm, nullptr) != ncclInvalidArgument) {
        std::fprintf(stderr, "FAIL: NCCL single-rank argument validation\n");
        return false;
    }
    if (ncclAllReduce(nullptr, nullptr, 0, ncclFloat32, ncclSum,
                      comm, nullptr) != ncclSuccess) {
        std::fprintf(stderr, "FAIL: zero-count NCCL collective should be a no-op\n");
        return false;
    }

    ncclComm_t comms[2] = {nullptr, nullptr};
    if (ncclCommInitAll(comms, 2, nullptr) != ncclInvalidArgument ||
        comms[0] != nullptr || comms[1] != nullptr) {
        std::fprintf(stderr, "FAIL: multi-device ncclCommInitAll partially initialized\n");
        return false;
    }
    int badDevice = 1;
    if (ncclCommInitAll(comms, 1, &badDevice) != ncclInvalidArgument || comms[0] != nullptr) {
        std::fprintf(stderr, "FAIL: unsupported NCCL device was accepted\n");
        return false;
    }
    if (ncclGroupEnd() != ncclInvalidUsage || ncclGroupStart() != ncclSuccess ||
        ncclGroupStart() != ncclInvalidUsage || ncclGroupEnd() != ncclSuccess) {
        std::fprintf(stderr, "FAIL: NCCL group state validation\n");
        return false;
    }
    const char* last = ncclGetLastError(comm);
    if (!last || std::strlen(last) == 0) {
        std::fprintf(stderr, "FAIL: NCCL valid communicator has no last-error string\n");
        return false;
    }

    ncclCommDestroy(comm);
    return true;
}

int main() {
    if (!test_version()) return 1;
    if (!test_comm_lifecycle()) return 1;
    if (!test_allreduce_identity()) return 1;
    if (!test_broadcast_identity()) return 1;
    if (!test_error_string()) return 1;
    if (!test_multi_rank_rejected()) return 1;
    if (!test_single_rank_validation()) return 1;

    std::printf("PASS: NCCL API tests\n");
    return 0;
}
