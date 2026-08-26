// Regression test for __cudaRegisterVar mapping a __device__/__constant__ symbol
// onto its device-side *name string* instead of its host shadow.
//
// clang and nvcc both emit
//     __cudaRegisterVar(handle, (char *)&Var, "Var", "Var", ext, size, constant, 0)
// -- the third argument is the name, not an address. CuMetal used to register that
// pointer as the symbol's device address, so cudaMemcpyToSymbol memcpy'd the caller's
// bytes straight over a string literal: SIGBUS when the constant pool sat in a
// read-only page, silent corruption of the binary's own data when it did not.
//
// The array is deliberately large (27904 B, matching cuda-samples/LargeKernelParameter,
// the case that first crashed) so a stray write lands well outside any one string.
//
// Host-side symbol APIs only -- no launch -- so this gates on the runtime, not on
// whether a given kernel can be lowered.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#define CONST_ELEMS 6976
#define DEV_ELEMS   256

__constant__ int const_params[CONST_ELEMS];
__device__ int dev_params[DEV_ELEMS];

// A kernel must exist for the translation unit to carry a fatbin and reach
// __cudaRegisterVar at all. It is never launched.
__global__ void touch_symbols(int *out) {
    *out = const_params[0] + dev_params[0];
}

static const char *kProbeName = "const_params";

static int check(const char *what, cudaError_t status) {
    if (status != cudaSuccess) {
        std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(status));
        return 1;
    }
    return 0;
}

int main() {
    // Snapshot the symbol names before touching anything. If the runtime writes
    // through the registered "device address" it scribbles over exactly these.
    char probe_before[32];
    std::strncpy(probe_before, kProbeName, sizeof(probe_before) - 1);
    probe_before[sizeof(probe_before) - 1] = '\0';

    static int host_const[CONST_ELEMS];
    static int host_readback[CONST_ELEMS];
    for (int i = 0; i < CONST_ELEMS; ++i) {
        host_const[i] = i * 3 + 1;
    }

    if (check("cudaMemcpyToSymbol(const_params)",
              cudaMemcpyToSymbol(const_params, host_const, sizeof(host_const), 0,
                                 cudaMemcpyHostToDevice))) {
        return 1;
    }
    if (check("cudaMemcpyFromSymbol(const_params)",
              cudaMemcpyFromSymbol(host_readback, const_params, sizeof(host_readback), 0,
                                   cudaMemcpyDeviceToHost))) {
        return 1;
    }
    for (int i = 0; i < CONST_ELEMS; ++i) {
        if (host_readback[i] != host_const[i]) {
            std::printf("FAIL: const_params[%d] round-trip: expected %d got %d\n", i,
                        host_const[i], host_readback[i]);
            return 1;
        }
    }

    int host_dev[DEV_ELEMS];
    for (int i = 0; i < DEV_ELEMS; ++i) {
        host_dev[i] = 1000 - i;
    }
    if (check("cudaMemcpyToSymbol(dev_params)",
              cudaMemcpyToSymbol(dev_params, host_dev, sizeof(host_dev), 0,
                                 cudaMemcpyHostToDevice))) {
        return 1;
    }
    int dev_readback[DEV_ELEMS];
    if (check("cudaMemcpyFromSymbol(dev_params)",
              cudaMemcpyFromSymbol(dev_readback, dev_params, sizeof(dev_readback), 0,
                                   cudaMemcpyDeviceToHost))) {
        return 1;
    }
    for (int i = 0; i < DEV_ELEMS; ++i) {
        if (dev_readback[i] != host_dev[i]) {
            std::printf("FAIL: dev_params[%d] round-trip: expected %d got %d\n", i, host_dev[i],
                        dev_readback[i]);
            return 1;
        }
    }

    // An offset write must land inside the symbol, not past it.
    const int tail = 424242;
    if (check("cudaMemcpyToSymbol(offset)",
              cudaMemcpyToSymbol(const_params, &tail, sizeof(tail),
                                 (CONST_ELEMS - 1) * sizeof(int), cudaMemcpyHostToDevice))) {
        return 1;
    }
    int tail_readback = 0;
    if (check("cudaMemcpyFromSymbol(offset)",
              cudaMemcpyFromSymbol(&tail_readback, const_params, sizeof(tail_readback),
                                   (CONST_ELEMS - 1) * sizeof(int), cudaMemcpyDeviceToHost))) {
        return 1;
    }
    if (tail_readback != tail) {
        std::printf("FAIL: offset round-trip: expected %d got %d\n", tail, tail_readback);
        return 1;
    }

    // Writing one element past the end must be rejected, not clamped or ignored.
    if (cudaMemcpyToSymbol(const_params, &tail, sizeof(tail), sizeof(host_const),
                           cudaMemcpyHostToDevice) != cudaErrorInvalidValue) {
        std::printf("FAIL: expected cudaErrorInvalidValue writing past the symbol\n");
        return 1;
    }
    cudaGetLastError();

    // The decisive check. Anchor on the compile-time address of the shadow rather
    // than on whatever the runtime reports, because a round-trip through a wrongly
    // registered address is self-consistent: memcpyTo and memcpyFrom both read the
    // same wrong place and agree with each other. The bytes have to actually be in
    // the variable.
    if (std::memcmp(static_cast<const void *>(const_params), host_const,
                    (CONST_ELEMS - 1) * sizeof(int)) != 0) {
        std::printf("FAIL: const_params shadow does not hold the bytes written to it\n");
        return 1;
    }
    if (std::memcmp(static_cast<const void *>(dev_params), host_dev, sizeof(host_dev)) != 0) {
        std::printf("FAIL: dev_params shadow does not hold the bytes written to it\n");
        return 1;
    }

    // cudaGetSymbolAddress must agree with where cudaMemcpyToSymbol wrote.
    void *addr = nullptr;
    if (check("cudaGetSymbolAddress", cudaGetSymbolAddress(&addr, const_params))) {
        return 1;
    }
    if (addr != static_cast<void *>(const_params)) {
        std::printf("FAIL: cudaGetSymbolAddress returned %p, expected %p\n", addr,
                    static_cast<void *>(const_params));
        return 1;
    }

    if (std::strcmp(kProbeName, probe_before) != 0) {
        std::printf("FAIL: symbol name string was overwritten ('%s' -> '%s')\n", probe_before,
                    kProbeName);
        return 1;
    }

    (void)touch_symbols;
    std::printf("PASS: constant/device symbol registration round-trips\n");
    return 0;
}
