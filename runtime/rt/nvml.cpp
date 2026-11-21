#include "nvml.h"
#include "cumetal_native.h"
#include "metal_backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

// NVML shim over the Metal device. Single device (index 0 only).
//
// Every figure comes from the MTLDevice through the backend boundary, never
// from host-wide statistics: unified memory makes "GPU memory" a budget, and
// reporting host RAM would mislead tools that size work from NVML.
//   name   MTLDevice.name, e.g. "Apple M4 Pro"
//   uuid   "GPU-cumetal-" + MTLDevice.registryID in hex. Stable while the
//          device stays registered; not a hardware serial.
//   total  MTLDevice.recommendedMaxWorkingSetSize
//   used   MTLDevice.currentAllocatedSize (this process's Metal allocations)
//   free   total - used, clamped at 0
// A missing figure is NVML_ERROR_NOT_SUPPORTED, never an estimate.

namespace {

std::atomic<unsigned int> g_init_count{0};
std::mutex g_init_mutex;

struct DeviceInfo {
    char name[64];
    char uuid[80];
    bool available = false;
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
    g_device = DeviceInfo{};
    cumetal::metal_backend::DeviceMemoryInfo info;
    std::string error;
    if (cumetal::metal_backend::query_device_memory(&info, &error) != cudaSuccess ||
        info.name.empty()) {
        return;
    }
    snprintf(g_device.name, sizeof(g_device.name), "%s", info.name.c_str());
    snprintf(g_device.uuid, sizeof(g_device.uuid), "GPU-cumetal-%016llx",
             static_cast<unsigned long long>(info.registry_id));
    g_device.available = true;
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
    if (!g_device.available) return NVML_ERROR_NOT_SUPPORTED;
    return copy_string(g_device.name, name, length);
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device)) return NVML_ERROR_INVALID_ARGUMENT;
    if (!g_device.available) return NVML_ERROR_NOT_SUPPORTED;
    return copy_string(g_device.uuid, uuid, length);
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (!initialized()) return NVML_ERROR_UNINITIALIZED;
    if (!valid_device(device) || !memory) return NVML_ERROR_INVALID_ARGUMENT;

    cumetal::metal_backend::DeviceMemoryInfo info;
    std::string error;
    if (cumetal::metal_backend::query_device_memory(&info, &error) != cudaSuccess ||
        info.recommended_working_set_bytes == 0) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    memory->total = info.recommended_working_set_bytes;
    memory->used = info.current_allocated_bytes < memory->total ? info.current_allocated_bytes
                                                               : memory->total;
    memory->free = memory->total - memory->used;
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
