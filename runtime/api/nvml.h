#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nvmlReturn_t {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_UNKNOWN = 999,
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef enum nvmlTemperatureSensors_t {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef enum nvmlClockType_t {
    NVML_CLOCK_GRAPHICS = 0,
    NVML_CLOCK_SM = 1,
    NVML_CLOCK_MEM = 2,
} nvmlClockType_t;

#define NVML_DEVICE_NAME_BUFFER_SIZE 64
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80

nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
const char* nvmlErrorString(nvmlReturn_t result);

nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device);

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length);

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory);
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization);
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                       nvmlTemperatureSensors_t sensorType,
                                       unsigned int* temp);
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power);
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device,
                                     nvmlClockType_t type,
                                     unsigned int* clock);

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);
nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length);

#ifdef __cplusplus
}
#endif
