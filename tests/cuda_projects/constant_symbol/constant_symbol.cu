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
// The final launch reads well past Metal's 4 KB setBytes limit, proving that the
// registered constant is bound as a real read-only buffer rather than an inline
// byte argument.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#define CONST_ELEMS 6976
#define DEV_ELEMS   256

__constant__ int const_params[CONST_ELEMS];
__device__ int dev_params[DEV_ELEMS];

__global__ void read_const_symbol(int *out) {
    *out = const_params[0] + const_params[4096];
}

__global__ void increment_device_symbol(int *out) {
    dev_params[7] += 3;
    *out = dev_params[7];
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
    // cudaGetSymbolAddress must agree with where cudaMemcpyToSymbol wrote.
    void *addr = nullptr;
    if (check("cudaGetSymbolAddress", cudaGetSymbolAddress(&addr, const_params))) {
        return 1;
    }
    size_t symbol_size = 0;
    if (check("cudaGetSymbolSize(const_params)",
              cudaGetSymbolSize(&symbol_size, const_params)) ||
        symbol_size != sizeof(host_const)) {
        std::printf("FAIL: const_params size expected %zu got %zu\n",
                    sizeof(host_const), symbol_size);
        return 1;
    }
    if (check("cudaGetSymbolSize(dev_params)",
              cudaGetSymbolSize(&symbol_size, dev_params)) ||
        symbol_size != sizeof(host_dev)) {
        std::printf("FAIL: dev_params size expected %zu got %zu\n",
                    sizeof(host_dev), symbol_size);
        return 1;
    }
    void *device_symbol_address = nullptr;
    if (check("cudaGetSymbolAddress(dev_params)",
              cudaGetSymbolAddress(&device_symbol_address, dev_params)) ||
        device_symbol_address == nullptr ||
        std::memcmp(device_symbol_address, host_dev, sizeof(host_dev)) != 0) {
        std::printf("FAIL: dev_params persistent storage was not initialized from symbol bytes\n");
        return 1;
    }
    int unregistered_symbol = 0;
    if (cudaGetSymbolSize(nullptr, const_params) != cudaErrorInvalidValue ||
        cudaGetSymbolSize(&symbol_size, &unregistered_symbol) != cudaErrorInvalidValue) {
        std::printf("FAIL: cudaGetSymbolSize negative paths were not rejected\n");
        return 1;
    }
    cudaGetLastError();
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

    int *device_out = nullptr;
    if (check("cudaMalloc(output)", cudaMalloc(&device_out, sizeof(*device_out)))) {
        return 1;
    }
    read_const_symbol<<<1, 1>>>(device_out);
    if (check("read_const_symbol launch", cudaGetLastError()) ||
        check("read_const_symbol synchronize", cudaDeviceSynchronize())) {
        cudaFree(device_out);
        return 1;
    }
    int host_output = 0;
    if (check("read_const_symbol copyback",
              cudaMemcpy(&host_output, device_out, sizeof(host_output),
                         cudaMemcpyDeviceToHost))) {
        cudaFree(device_out);
        return 1;
    }
    cudaFree(device_out);
    const int expected_output = host_const[0] + host_const[4096];
    if (host_output != expected_output) {
        std::printf("FAIL: constant kernel expected %d got %d\n", expected_output,
                    host_output);
        return 1;
    }

    if (check("cudaMalloc(device global output)",
              cudaMalloc(&device_out, sizeof(*device_out)))) {
        return 1;
    }
    increment_device_symbol<<<1, 1>>>(device_out);
    increment_device_symbol<<<1, 1>>>(device_out);
    if (check("increment_device_symbol launch", cudaGetLastError()) ||
        check("increment_device_symbol synchronize", cudaDeviceSynchronize())) {
        cudaFree(device_out);
        return 1;
    }
    host_output = 0;
    if (check("increment_device_symbol copyback",
              cudaMemcpy(&host_output, device_out, sizeof(host_output),
                         cudaMemcpyDeviceToHost))) {
        cudaFree(device_out);
        return 1;
    }
    cudaFree(device_out);
    const int expected_device_value = host_dev[7] + 6;
    if (host_output != expected_device_value) {
        std::printf("FAIL: persistent device global expected %d got %d\n",
                    expected_device_value, host_output);
        return 1;
    }
    if (check("cudaMemcpyFromSymbol(dev_params after kernels)",
              cudaMemcpyFromSymbol(dev_readback, dev_params, sizeof(dev_readback), 0,
                                   cudaMemcpyDeviceToHost)) ||
        dev_readback[7] != expected_device_value) {
        std::printf("FAIL: device global copyback expected %d got %d\n",
                    expected_device_value, dev_readback[7]);
        return 1;
    }

    std::printf("PASS: constant/device symbol registration and persistent kernel binding validated\n");
    return 0;
}
