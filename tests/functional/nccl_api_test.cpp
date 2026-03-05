#include "nccl.h"

#include <cstdio>
#include <cstring>

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

int main() {
    if (!test_version()) return 1;
    if (!test_comm_lifecycle()) return 1;
    if (!test_allreduce_identity()) return 1;
    if (!test_broadcast_identity()) return 1;
    if (!test_error_string()) return 1;
    if (!test_multi_rank_rejected()) return 1;

    std::printf("PASS: NCCL API tests\n");
    return 0;
}
