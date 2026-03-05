#include <nvtx3/nvToolsExt.h>
#include <nvToolsExt.h>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_range_push_pop() {
    int depth = nvtxRangePushA("test_range");
    CHECK(depth == 0, "nvtxRangePushA returns 0");
    int depth2 = nvtxRangePop();
    CHECK(depth2 == 0, "nvtxRangePop returns 0");
}

static void test_range_start_end() {
    nvtxRangeId_t id = nvtxRangeStartA("test_range");
    CHECK(id == 0, "nvtxRangeStartA returns 0");
    nvtxRangeEnd(id);
    CHECK(true, "nvtxRangeEnd no crash");
}

static void test_mark() {
    nvtxMarkA("test_mark");
    CHECK(true, "nvtxMarkA no crash");
}

static void test_event_attributes() {
    nvtxEventAttributes_t attribs = {};
    attribs.version = NVTX_VERSION;
    attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attribs.colorType = NVTX_COLOR_ARGB;
    attribs.color = 0xFF00FF00;
    attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attribs.message.ascii = "test";

    nvtxRangeId_t id = nvtxRangeStartEx(&attribs);
    CHECK(id == 0, "nvtxRangeStartEx with attribs");
    nvtxRangeEnd(id);
    CHECK(true, "nvtxRangeEnd with attribs");

    int depth = nvtxRangePushEx(&attribs);
    CHECK(depth == 0, "nvtxRangePushEx");
    nvtxRangePop();

    nvtxMarkEx(&attribs);
    CHECK(true, "nvtxMarkEx no crash");
}

static void test_domain_api() {
    nvtxDomainHandle_t domain = nvtxDomainCreateA("test_domain");
    CHECK(domain == 0, "nvtxDomainCreateA returns null (no-op)");

    nvtxEventAttributes_t attribs = {};
    attribs.version = NVTX_VERSION;
    attribs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attribs.message.ascii = "domain_test";

    nvtxDomainRangePushEx(domain, &attribs);
    nvtxDomainRangePop(domain);
    nvtxDomainMarkEx(domain, &attribs);
    nvtxDomainDestroy(domain);
    CHECK(true, "domain API no crash");
}

static void test_naming() {
    nvtxNameOsThreadA(0, "main_thread");
    nvtxNameCuDeviceA(0, "gpu0");
    nvtxNameCuStreamA(nullptr, "default_stream");
    CHECK(true, "naming API no crash");
}

int main() {
    test_range_push_pop();
    test_range_start_end();
    test_mark();
    test_event_attributes();
    test_domain_api();
    test_naming();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
