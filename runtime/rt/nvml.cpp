#include "nvml.h"
#include "cumetal_native.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mach/mach.h>
#include <mutex>
#include <sys/sysctl.h>

// NVML shim: reports Apple Silicon GPU info via sysctl/mach APIs.
// Single device (device index 0 only).

namespace {

std::atomic<unsigned int> g_init_count{0};
std::mutex g_init_mutex;

struct DeviceInfo {
    char name[64];
    char uuid[80];
    unsigned long long total_mem;
};

static DeviceInfo g_device;

bool initialized() {
    return g_init_count.load(std::memory_order_acquire) != 0;
}

bool valid_device(nvmlDevice_t device) {
    return device == reinterpret_cast<nvmlDevice_t>(1);
}

nvmlReturn_t copy_string(const char* source, char* destination, unsigned int length) {
    if (!source || !destination || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    const size_t required = std::strlen(source) + 1;
    if (required > length) {
        destination[0] = '\0';
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    std::memcpy(destination, source, required);
    return NVML_SUCCESS;
}

void init_device_info() {
    // Get chip name via sysctl
    char brand[128] = {};
    size_t brand_len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, nullptr, 0) == 0) {
        snprintf(g_device.name, sizeof(g_device.name), "Apple Silicon (%s)", brand);
    } else {
        snprintf(g_device.name, sizeof(g_device.name), "Apple Silicon GPU");
    }

    // Synthetic UUID
    snprintf(g_device.uuid, sizeof(g_device.uuid),
             "GPU-cumetal-apple-silicon-0000-000000000000");

    // Total physical memory
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
    g_device.total_mem = memsize;
}

} // namespace

extern "C" {

nvmlReturn_t nvmlInit(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    unsigned int count = g_init_count.load(std::memory_order_relaxed);
    if (count == 0) init_device_info();
    if (count == std::numeric_limits<unsigned int>::max()) return NVML_ERROR_UNKNOWN;
    g_init_count.store(count + 1, std::memory_order_release);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit_v2(void) { return nvmlInit(); }

nvmlReturn_t nvmlShutdown(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    unsigned int count = g_init_count.load(std::memory_order_relaxed);
    if (count == 0) return NVML_ERROR_UNINITIALIZED;
    g_init_count.store(count - 1, std::memory_order_release);
    return NVML_SUCCESS;
}

const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
        case NVML_ERROR_NO_PERMISSION: return "No Permission";
        case NVML_ERROR_ALREADY_INITIALIZED: return "Already Initialized";
        case NVML_ERROR_NOT_FOUND: return "Not Found";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
        default: return "Unknown Error";
    }
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    *deviceCount = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    if (index != 0) return NVML_ERROR_INVALID_ARGUMENT;
    // Use a sentinel pointer for device 0
    *device = reinterpret_cast<nvmlDevice_t>(1);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    return nvmlDeviceGetHandleByIndex(index, device);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device)) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(g_device.name, name, length);
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device)) return NVML_ERROR_INVALID_ARGUMENT;
    return copy_string(g_device.uuid, uuid, length);
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !memory) return NVML_ERROR_INVALID_ARGUMENT;

    memory->total = g_device.total_mem;

    // Approximate free memory via mach VM stats
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
        unsigned long long page_size = vm_page_size;
        unsigned long long free_pages = vm_stat.free_count + vm_stat.inactive_count;
        memory->free = free_pages * page_size;
        memory->used = memory->total > memory->free ? memory->total - memory->free : 0;
    } else {
        memory->free = memory->total / 2;
        memory->used = memory->total / 2;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !utilization) return NVML_ERROR_INVALID_ARGUMENT;
    // Public macOS APIs do not expose an NVML-equivalent utilization sample.
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                       nvmlTemperatureSensors_t /*sensorType*/,
                                       unsigned int* temp) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !temp) return NVML_ERROR_INVALID_ARGUMENT;
    // Temperature not easily available via public macOS APIs
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !power) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device,
                                     nvmlClockType_t /*type*/,
                                     unsigned int* clock) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !clock) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    return copy_string("cumetal-" CUMETAL_VERSION_STRING, version, length);
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    return copy_string("12.0", version, length);
}

} // extern "C"
