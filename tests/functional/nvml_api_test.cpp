#include "nvml.h"
#include "cumetal_native.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/sysctl.h>

static bool test_init_shutdown() {
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
        std::fprintf(stderr, "FAIL: nvmlInit returned %d\n", r);
        return false;
    }
    r = nvmlShutdown();
    if (r != NVML_SUCCESS) {
        std::fprintf(stderr, "FAIL: nvmlShutdown returned %d\n", r);
        return false;
    }
    return true;
}

static bool test_device_count() {
    nvmlInit();
    unsigned int count = 0;
    nvmlReturn_t r = nvmlDeviceGetCount(&count);
    if (r != NVML_SUCCESS || count != 1) {
        std::fprintf(stderr, "FAIL: device count=%u (expected 1)\n", count);
        nvmlShutdown();
        return false;
    }
    nvmlShutdown();
    return true;
}

static bool test_device_name() {
    nvmlInit();
    nvmlDevice_t dev = nullptr;
    nvmlDeviceGetHandleByIndex(0, &dev);

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    nvmlReturn_t r = nvmlDeviceGetName(dev, name, sizeof(name));
    if (r != NVML_SUCCESS || std::strlen(name) == 0) {
        std::fprintf(stderr, "FAIL: device name empty or error %d\n", r);
        nvmlShutdown();
        return false;
    }

    nvmlShutdown();
    return true;
}

static bool test_memory_info() {
    nvmlInit();
    nvmlDevice_t dev = nullptr;
    nvmlDeviceGetHandleByIndex(0, &dev);

    nvmlMemory_t mem = {};
    nvmlReturn_t r = nvmlDeviceGetMemoryInfo(dev, &mem);
    if (r != NVML_SUCCESS || mem.total == 0) {
        std::fprintf(stderr, "FAIL: memory info total=0 or error %d\n", r);
        nvmlShutdown();
        return false;
    }

    nvmlShutdown();
    return true;
}

static bool test_driver_version() {
    nvmlInit();
    char ver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    nvmlReturn_t r = nvmlSystemGetDriverVersion(ver, sizeof(ver));
    const char* expected = "cumetal-" CUMETAL_VERSION_STRING;
    if (r != NVML_SUCCESS || std::strcmp(ver, expected) != 0) {
        std::fprintf(stderr, "FAIL: driver version '%s', expected '%s' (error %d)\n",
                     ver, expected, r);
        nvmlShutdown();
        return false;
    }
    nvmlShutdown();
    return true;
}

static bool test_error_string() {
    const char* s = nvmlErrorString(NVML_SUCCESS);
    if (!s || std::strlen(s) == 0) {
        std::fprintf(stderr, "FAIL: nvmlErrorString returned null/empty\n");
        return false;
    }
    return true;
}

static bool test_uninitialized_guard() {
    // Calling without init should return UNINITIALIZED
    unsigned int count = 0;
    nvmlReturn_t r = nvmlDeviceGetCount(&count);
    if (r != NVML_ERROR_UNINITIALIZED) {
        std::fprintf(stderr, "FAIL: expected UNINITIALIZED, got %d\n", r);
        return false;
    }
    return true;
}

static bool test_validation_and_init_refcount() {
    if (nvmlInit() != NVML_SUCCESS || nvmlInit_v2() != NVML_SUCCESS) return false;
    if (nvmlShutdown() != NVML_SUCCESS) return false;

    unsigned int count = 0;
    if (nvmlDeviceGetCount(&count) != NVML_SUCCESS || count != 1) {
        std::fprintf(stderr, "FAIL: nested NVML init was not reference counted\n");
        return false;
    }

    nvmlDevice_t device = nullptr;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) return false;
    char tiny[2] = {'x', 'x'};
    if (nvmlDeviceGetName(device, tiny, sizeof(tiny)) != NVML_ERROR_INSUFFICIENT_SIZE ||
        tiny[0] != '\0') {
        std::fprintf(stderr, "FAIL: undersized NVML name buffer was silently truncated\n");
        return false;
    }
    nvmlDevice_t bogus = reinterpret_cast<nvmlDevice_t>(2);
    nvmlMemory_t memory = {};
    if (nvmlDeviceGetMemoryInfo(bogus, &memory) != NVML_ERROR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "FAIL: invalid NVML device handle was accepted\n");
        return false;
    }
    nvmlUtilization_t utilization = {77, 88};
    if (nvmlDeviceGetUtilizationRates(device, &utilization) != NVML_ERROR_NOT_SUPPORTED ||
        utilization.gpu != 77 || utilization.memory != 88) {
        std::fprintf(stderr, "FAIL: unavailable NVML utilization was reported as measured\n");
        return false;
    }

    if (nvmlShutdown() != NVML_SUCCESS || nvmlShutdown() != NVML_ERROR_UNINITIALIZED) {
        std::fprintf(stderr, "FAIL: NVML shutdown reference-count validation\n");
        return false;
    }
    return true;
}

// #214: memory figures come from the Metal device, not host RAM. total is
// the device's recommended working set (what cudaMemGetInfo also reports),
// strictly below physical memory, and used follows this process's Metal
// allocations.
static bool test_memory_is_metal_device_memory() {
    if (nvmlInit() != NVML_SUCCESS) return false;
    nvmlDevice_t dev = nullptr;
    nvmlDeviceGetHandleByIndex(0, &dev);

    size_t cuda_free = 0, cuda_total = 0;
    if (cudaMemGetInfo(&cuda_free, &cuda_total) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemGetInfo failed\n");
        return false;
    }
    std::uint64_t host_ram = 0;
    size_t len = sizeof(host_ram);
    sysctlbyname("hw.memsize", &host_ram, &len, nullptr, 0);

    nvmlMemory_t before = {};
    if (nvmlDeviceGetMemoryInfo(dev, &before) != NVML_SUCCESS) {
        std::fprintf(stderr, "FAIL: NVML memory info failed\n");
        return false;
    }
    if (before.total != cuda_total || before.total >= host_ram ||
        before.free + before.used != before.total) {
        std::fprintf(stderr,
                     "FAIL: NVML total=%llu free=%llu used=%llu (cuda total=%zu host=%llu)\n",
                     before.total, before.free, before.used, cuda_total,
                     static_cast<unsigned long long>(host_ram));
        return false;
    }

    constexpr size_t kBytes = 64u << 20;
    void* block = nullptr;
    if (cudaMalloc(&block, kBytes) != cudaSuccess) return false;
    nvmlMemory_t after = {};
    nvmlDeviceGetMemoryInfo(dev, &after);
    cudaFree(block);
    if (after.used < before.used + kBytes || after.free + after.used != after.total) {
        std::fprintf(stderr, "FAIL: NVML used did not follow a 64 MiB allocation (%llu -> %llu)\n",
                     before.used, after.used);
        return false;
    }

    // Name is the Metal device's, matching the CUDA device properties.
    cudaDeviceProp prop = {};
    cudaGetDeviceProperties(&prop, 0);
    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    char uuid[96] = {};
    if (nvmlDeviceGetName(dev, name, sizeof(name)) != NVML_SUCCESS ||
        std::strcmp(name, prop.name) != 0 ||
        nvmlDeviceGetUUID(dev, uuid, sizeof(uuid)) != NVML_SUCCESS ||
        std::strncmp(uuid, "GPU-cumetal-", 12) != 0 || std::strlen(uuid) != 12 + 16) {
        std::fprintf(stderr, "FAIL: NVML name '%s' (cuda '%s') uuid '%s'\n", name, prop.name,
                     uuid);
        return false;
    }
    if (nvmlDeviceGetMemoryInfo(dev, nullptr) != NVML_ERROR_INVALID_ARGUMENT) {
        std::fprintf(stderr, "FAIL: null NVML memory output accepted\n");
        return false;
    }
    nvmlShutdown();
    nvmlMemory_t uninit = {};
    if (nvmlDeviceGetMemoryInfo(dev, &uninit) != NVML_ERROR_UNINITIALIZED) {
        std::fprintf(stderr, "FAIL: NVML memory info answered after shutdown\n");
        return false;
    }
    return true;
}

int main() {
    if (!test_uninitialized_guard()) return 1;
    if (!test_init_shutdown()) return 1;
    if (!test_device_count()) return 1;
    if (!test_device_name()) return 1;
    if (!test_memory_info()) return 1;
    if (!test_driver_version()) return 1;
    if (!test_error_string()) return 1;
    if (!test_validation_and_init_refcount()) return 1;
    if (!test_memory_is_metal_device_memory()) return 1;

    std::printf("PASS: NVML API tests\n");
    return 0;
}
