#include "cuda.h"
#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: runtime device activation failed\n");
        return 1;
    }
    CUdevice device = -1;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS || device != 0) {
        std::fprintf(stderr, "FAIL: runtime did not activate the primary Driver context\n");
        return 1;
    }

    int vmm = 0;
    int compression = 0;
    if (cuDeviceGetAttribute(
            &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(
            &compression, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
            device) != CUDA_SUCCESS ||
        vmm != 1 || compression != 1) {
        std::fprintf(stderr, "FAIL: Driver VMM capability attributes\n");
        return 1;
    }

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS ||
        granularity == 0 || (granularity & (granularity - 1)) != 0) {
        std::fprintf(stderr, "FAIL: VMM allocation granularity\n");
        return 1;
    }
    if (cuMemGetAllocationGranularity(
            nullptr, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null granularity output was accepted\n");
        return 1;
    }

    CUdeviceptr address = 0;
    if (cuMemAddressReserve(&address, granularity, 0, 0, 0) != CUDA_SUCCESS ||
        address == 0) {
        std::fprintf(stderr, "FAIL: VMM address reservation\n");
        return 1;
    }
    if (cuMemAddressReserve(nullptr, granularity, 0, 0, 0) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuMemAddressReserve(&address, granularity, 3, 0, 0) !=
            CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: invalid VMM reservations were accepted\n");
        return 1;
    }

    CUmemGenericAllocationHandle handle = 0;
    if (cuMemCreate(&handle, granularity, &prop, 0) != CUDA_SUCCESS ||
        handle == 0) {
        std::fprintf(stderr, "FAIL: VMM physical allocation handle\n");
        return 1;
    }
    CUmemAllocationProp observed{};
    if (cuMemGetAllocationPropertiesFromHandle(&observed, handle) !=
            CUDA_SUCCESS ||
        observed.allocFlags.compressionType !=
            CU_MEM_ALLOCATION_COMP_GENERIC) {
        std::fprintf(stderr, "FAIL: compression hint did not round-trip\n");
        return 1;
    }
    if (cuMemMap(address, granularity, 0, handle, 0) != CUDA_SUCCESS ||
        cuMemMap(address, granularity, 0, handle, 0) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuMemAddressFree(address, granularity) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: VMM map lifecycle\n");
        return 1;
    }

    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(address, granularity, &access, 1) != CUDA_SUCCESS ||
        cuMemSetAccess(address, granularity, nullptr, 1) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuMemFree(address) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: VMM access/free lifecycle\n");
        return 1;
    }

    void* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(address));
    unsigned char host[16] = {};
    if (cudaMemset(pointer, 0x5a, sizeof(host)) != cudaSuccess ||
        cudaMemcpy(host, pointer, sizeof(host), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: mapped VMM allocation is not runtime-usable\n");
        return 1;
    }
    for (unsigned char byte : host) {
        if (byte != 0x5a) {
            std::fprintf(stderr, "FAIL: mapped VMM allocation data mismatch\n");
            return 1;
        }
    }

    if (cuMemRelease(handle) != CUDA_SUCCESS ||
        cuMemGetAllocationPropertiesFromHandle(&observed, handle) !=
            CUDA_ERROR_INVALID_VALUE ||
        cuMemRelease(handle) != CUDA_ERROR_INVALID_VALUE ||
        cuMemUnmap(address, granularity) != CUDA_SUCCESS ||
        cuMemUnmap(address, granularity) != CUDA_ERROR_INVALID_VALUE ||
        cuMemAddressFree(address, granularity) != CUDA_SUCCESS ||
        cuMemAddressFree(address, granularity) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: VMM release/unmap/address-free lifecycle\n");
        return 1;
    }

    std::printf("PASS: Driver VMM lifecycle and compression hint compatibility\n");
    return 0;
}
