#pragma once
// CuMetal NVTX shim: no-op stubs for NVIDIA Tools Extension Library.
// Profiling annotations are silently ignored on Metal.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Range handle type
typedef uint64_t nvtxRangeId_t;

// Event attributes
typedef struct nvtxEventAttributes_v2 {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t  colorType;
    uint32_t color;
    int32_t  payloadType;
    int32_t  reserved0;
    union {
        uint64_t ullValue;
        int64_t  llValue;
        double   dValue;
    } payload;
    int32_t  messageType;
    union {
        const char*    ascii;
        const wchar_t* unicode;
    } message;
} nvtxEventAttributes_t;

// Color types
#define NVTX_COLOR_UNKNOWN  0
#define NVTX_COLOR_ARGB     1

// Message types
#define NVTX_MESSAGE_UNKNOWN       0
#define NVTX_MESSAGE_TYPE_ASCII    1
#define NVTX_MESSAGE_TYPE_UNICODE  2

// Payload types
#define NVTX_PAYLOAD_UNKNOWN             0
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 1
#define NVTX_PAYLOAD_TYPE_INT64          2
#define NVTX_PAYLOAD_TYPE_DOUBLE         3

// Version
#define NVTX_VERSION 2
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE sizeof(nvtxEventAttributes_t)

// No-op API stubs
static inline void nvtxMarkA(const char* message) { (void)message; }
static inline void nvtxMarkW(const wchar_t* message) { (void)message; }
static inline void nvtxMarkEx(const nvtxEventAttributes_t* attribs) { (void)attribs; }

static inline nvtxRangeId_t nvtxRangeStartA(const char* message) { (void)message; return 0; }
static inline nvtxRangeId_t nvtxRangeStartW(const wchar_t* message) { (void)message; return 0; }
static inline nvtxRangeId_t nvtxRangeStartEx(const nvtxEventAttributes_t* attribs) { (void)attribs; return 0; }
static inline void nvtxRangeEnd(nvtxRangeId_t id) { (void)id; }

static inline int nvtxRangePushA(const char* message) { (void)message; return 0; }
static inline int nvtxRangePushW(const wchar_t* message) { (void)message; return 0; }
static inline int nvtxRangePushEx(const nvtxEventAttributes_t* attribs) { (void)attribs; return 0; }
static inline int nvtxRangePop(void) { return 0; }

static inline void nvtxNameOsThreadA(uint32_t threadId, const char* name) { (void)threadId; (void)name; }
static inline void nvtxNameCuDeviceA(int device, const char* name) { (void)device; (void)name; }
static inline void nvtxNameCuStreamA(void* stream, const char* name) { (void)stream; (void)name; }

// Domain API (NVTX3)
typedef struct nvtxDomainHandle* nvtxDomainHandle_t;
static inline nvtxDomainHandle_t nvtxDomainCreateA(const char* name) { (void)name; return 0; }
static inline void nvtxDomainDestroy(nvtxDomainHandle_t domain) { (void)domain; }
static inline nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attribs) {
    (void)domain; (void)attribs; return 0;
}
static inline void nvtxDomainRangeEnd(nvtxDomainHandle_t domain, nvtxRangeId_t id) {
    (void)domain; (void)id;
}
static inline int nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attribs) {
    (void)domain; (void)attribs; return 0;
}
static inline int nvtxDomainRangePop(nvtxDomainHandle_t domain) { (void)domain; return 0; }
static inline void nvtxDomainMarkEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attribs) {
    (void)domain; (void)attribs;
}

#ifdef __cplusplus
}
#endif
