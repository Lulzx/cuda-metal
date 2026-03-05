#include "nvml.h"

#include <cstdio>
#include <cstring>

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
    if (r != NVML_SUCCESS || std::strlen(ver) == 0) {
        std::fprintf(stderr, "FAIL: driver version empty or error %d\n", r);
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

int main() {
    if (!test_uninitialized_guard()) return 1;
    if (!test_init_shutdown()) return 1;
    if (!test_device_count()) return 1;
    if (!test_device_name()) return 1;
    if (!test_memory_info()) return 1;
    if (!test_driver_version()) return 1;
    if (!test_error_string()) return 1;

    std::printf("PASS: NVML API tests\n");
    return 0;
}
