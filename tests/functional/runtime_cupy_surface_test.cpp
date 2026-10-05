// Runtime, driver, cuFFT Xt and NCCL entry points CuPy binds at build time.
// Each one either does its work or refuses with an error code; none may
// report success for work it skipped.

#include "cuda.h"
#include "cuda_runtime.h"
#include "cufftXt.h"
#include "nccl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, ...)                          \
    do {                                          \
        if (!(cond)) {                            \
            std::fprintf(stderr, "FAIL: ");       \
            std::fprintf(stderr, __VA_ARGS__);    \
            std::fprintf(stderr, "\n");           \
            return false;                         \
        }                                         \
    } while (0)

static bool test_pci_bus_id() {
    char bus[32] = {};
    CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 0) == cudaSuccess, "GetPCIBusId");
    CHECK(std::strcmp(bus, "0000:00:00.0") == 0, "bus id is %s", bus);
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "GetDeviceProperties");
    CHECK(prop.pciBusID == 0 && prop.pciDeviceID == 0 && prop.pciDomainID == 0,
          "properties disagree with the bus id string");
    char tiny[4] = {};
    CHECK(cudaDeviceGetPCIBusId(tiny, sizeof(tiny), 0) == cudaErrorInvalidValue,
          "short buffer accepted");
    CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 1) == cudaErrorInvalidDevice,
          "device 1 accepted");

    int device = -1;
    CHECK(cudaDeviceGetByPCIBusId(&device, bus) == cudaSuccess && device == 0,
          "round trip of %s", bus);
    device = -1;
    CHECK(cudaDeviceGetByPCIBusId(&device, "00:00.0") == cudaSuccess && device == 0,
          "domain-less form");
    CHECK(cudaDeviceGetByPCIBusId(&device, "0000:01:00.0") == cudaErrorInvalidDevice,
          "foreign bus accepted");
    CHECK(cudaDeviceGetByPCIBusId(&device, "not-a-bus") == cudaErrorInvalidValue,
          "garbage accepted");
    cudaGetLastError();
    return true;
}

static bool test_stream_priority() {
    int least = 1, greatest = 1;
    CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess, "range");
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest) ==
              cudaSuccess,
          "create with priority");
    int priority = 42;
    CHECK(cudaStreamGetPriority(stream, &priority) == cudaSuccess, "GetPriority");
    CHECK(priority >= greatest && priority <= least, "priority %d outside [%d,%d]", priority,
          greatest, least);
    priority = 42;
    CHECK(cudaStreamGetPriority(nullptr, &priority) == cudaSuccess && priority == 0,
          "legacy stream priority");
    CHECK(cudaStreamGetPriority(stream, nullptr) == cudaErrorInvalidValue, "null out");
    CHECK(cudaStreamDestroy(stream) == cudaSuccess, "destroy");
    cudaGetLastError();
    return true;
}

static bool test_mem_pool() {
    cudaMemPool_t def = nullptr, cur = nullptr;
    CHECK(cudaDeviceGetDefaultMemPool(&def, 0) == cudaSuccess && def != nullptr, "default");
    CHECK(cudaDeviceGetMemPool(&cur, 0) == cudaSuccess && cur == def, "current == default");

    cudaMemPoolProps props{};
    cudaMemPool_t pool = nullptr;
    CHECK(cudaMemPoolCreate(&pool, &props) == cudaSuccess && pool != nullptr, "create");
    CHECK(cudaDeviceSetMemPool(0, pool) == cudaSuccess, "set");
    CHECK(cudaDeviceGetMemPool(&cur, 0) == cudaSuccess && cur == pool,
          "GetMemPool does not return the installed pool");
    CHECK(cudaMemPoolTrimTo(pool, 0) == cudaSuccess, "trim");
    CHECK(cudaDeviceSetMemPool(0, def) == cudaSuccess, "restore");
    CHECK(cudaMemPoolDestroy(pool) == cudaSuccess, "destroy");

    CHECK(cudaMemPoolTrimTo(nullptr, 0) == cudaErrorInvalidValue, "trim null");
    CHECK(cudaDeviceGetMemPool(&cur, 1) == cudaErrorInvalidDevice, "device 1");
    CHECK(cudaDeviceSetMemPool(0, nullptr) == cudaErrorInvalidValue, "set null");
    cudaGetLastError();
    return true;
}

static bool test_ipc_refuses() {
    void* ptr = nullptr;
    CHECK(cudaMalloc(&ptr, 256) == cudaSuccess, "malloc");
    cudaIpcMemHandle_t mem{};
    CHECK(cudaIpcGetMemHandle(&mem, ptr) == cudaErrorNotSupported, "GetMemHandle");
    void* opened = nullptr;
    CHECK(cudaIpcOpenMemHandle(&opened, mem, cudaIpcMemLazyEnablePeerAccess) ==
              cudaErrorNotSupported,
          "OpenMemHandle");
    CHECK(opened == nullptr, "OpenMemHandle produced a pointer");
    CHECK(cudaIpcCloseMemHandle(ptr) == cudaErrorNotSupported, "CloseMemHandle");
    cudaEvent_t event = nullptr;
    CHECK(cudaEventCreate(&event) == cudaSuccess, "event");
    cudaIpcEventHandle_t ev{};
    CHECK(cudaIpcGetEventHandle(&ev, event) == cudaErrorNotSupported, "GetEventHandle");
    cudaEvent_t other = nullptr;
    CHECK(cudaIpcOpenEventHandle(&other, ev) == cudaErrorNotSupported, "OpenEventHandle");
    static_assert(sizeof(cudaIpcMemHandle_t) == 64, "CUDA IPC handles are 64 bytes");
    CHECK(cudaEventDestroy(event) == cudaSuccess, "event destroy");
    CHECK(cudaFree(ptr) == cudaSuccess, "free");
    cudaGetLastError();
    return true;
}

static bool test_array_async_copies() {
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t array = nullptr;
    const size_t w = 8, h = 4;
    CHECK(cudaMallocArray(&array, &desc, w, h, 0) == cudaSuccess, "MallocArray");

    cudaChannelFormatDesc got{};
    CHECK(cudaGetChannelDesc(&got, array) == cudaSuccess, "GetChannelDesc");
    CHECK(got.x == 32 && got.y == 0 && got.f == cudaChannelFormatKindFloat,
          "channel desc x=%d f=%d", got.x, static_cast<int>(got.f));
    CHECK(cudaGetChannelDesc(nullptr, array) == cudaErrorInvalidValue, "null desc");

    std::vector<float> in(w * h), out(w * h, -1.0f);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i) * 0.5f;
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess, "stream");
    CHECK(cudaMemcpy2DToArrayAsync(array, 0, 0, in.data(), w * sizeof(float), w * sizeof(float),
                                   h, cudaMemcpyHostToDevice, stream) == cudaSuccess,
          "ToArrayAsync");
    CHECK(cudaMemcpy2DFromArrayAsync(out.data(), w * sizeof(float), array, 0, 0,
                                     w * sizeof(float), h, cudaMemcpyDeviceToHost,
                                     stream) == cudaSuccess,
          "FromArrayAsync");
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "sync");
    for (size_t i = 0; i < in.size(); ++i) {
        CHECK(out[i] == in[i], "element %zu: %f != %f", i, out[i], in[i]);
    }
    // A sub-rectangle at an offset lands where the offsets say.
    std::vector<float> patch = {100.0f, 101.0f};
    CHECK(cudaMemcpy2DToArrayAsync(array, 2 * sizeof(float), 1, patch.data(),
                                   2 * sizeof(float), 2 * sizeof(float), 1,
                                   cudaMemcpyHostToDevice, stream) == cudaSuccess,
          "offset copy");
    float pair[2] = {};
    CHECK(cudaMemcpy2DFromArrayAsync(pair, 2 * sizeof(float), array, 2 * sizeof(float), 1,
                                     2 * sizeof(float), 1, cudaMemcpyDeviceToHost,
                                     stream) == cudaSuccess,
          "offset read");
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "sync 2");
    CHECK(pair[0] == 100.0f && pair[1] == 101.0f, "offset patch read %f %f", pair[0], pair[1]);
    CHECK(cudaMemcpy2DToArrayAsync(array, 0, h, in.data(), w * sizeof(float),
                                   w * sizeof(float), 1, cudaMemcpyHostToDevice,
                                   stream) == cudaErrorInvalidValue,
          "out-of-range row accepted");
    CHECK(cudaStreamDestroy(stream) == cudaSuccess, "stream destroy");
    CHECK(cudaFreeArray(array) == cudaSuccess, "FreeArray");
    cudaGetLastError();
    return true;
}

static bool test_link_single_image() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit");
    CUlinkState state = nullptr;
    CHECK(cuLinkCreate(0, nullptr, nullptr, &state) == CUDA_SUCCESS && state != nullptr,
          "LinkCreate");
    void* out = nullptr;
    size_t size = 0;
    CHECK(cuLinkComplete(state, &out, &size) == CUDA_ERROR_INVALID_VALUE,
          "empty link completed");
    static const char kPtx[] = ".version 7.0\n.target sm_80\n.address_size 64\n"
                               ".visible .entry k() { ret; }\n";
    CHECK(cuLinkAddData(state, CU_JIT_INPUT_PTX, const_cast<char*>(kPtx), sizeof(kPtx) - 1,
                        "k.ptx", 0, nullptr, nullptr) == CUDA_SUCCESS,
          "AddData PTX");
    CHECK(cuLinkAddData(state, CU_JIT_INPUT_PTX, const_cast<char*>(kPtx), sizeof(kPtx) - 1,
                        "k2.ptx", 0, nullptr, nullptr) == CUDA_ERROR_NOT_SUPPORTED,
          "a second image was accepted without a device linker");
    CHECK(cuLinkComplete(state, &out, &size) == CUDA_SUCCESS, "LinkComplete");
    CHECK(out != nullptr && size == sizeof(kPtx), "linked image size %zu", size);
    CHECK(std::memcmp(out, kPtx, sizeof(kPtx)) == 0, "linked image differs from its input");
    CHECK(cuLinkDestroy(state) == CUDA_SUCCESS, "LinkDestroy");

    CHECK(cuLinkCreate(0, nullptr, nullptr, &state) == CUDA_SUCCESS, "LinkCreate 2");
    char obj[16] = {1};
    CHECK(cuLinkAddData(state, CU_JIT_INPUT_LIBRARY, obj, sizeof(obj), "lib", 0, nullptr,
                        nullptr) == CUDA_ERROR_NOT_SUPPORTED,
          "library input accepted");
    CHECK(cuLinkAddFile(state, CU_JIT_INPUT_CUBIN, "/nonexistent/cumetal.cubin", 0, nullptr,
                        nullptr) == CUDA_ERROR_FILE_NOT_FOUND,
          "missing file");
    CHECK(cuLinkDestroy(state) == CUDA_SUCCESS, "LinkDestroy 2");
    CHECK(cuLinkDestroy(nullptr) == CUDA_ERROR_INVALID_HANDLE, "destroy null");
    return true;
}

static bool test_cufft_xt() {
    cufftHandle plan = 0;
    CHECK(cufftCreate(&plan) == CUFFT_SUCCESS, "create");
    CHECK(cufftSetAutoAllocation(plan, 0) == CUFFT_SUCCESS, "SetAutoAllocation");
    long long n = 8;
    size_t work = 0;
    CHECK(cufftXtMakePlanMany(plan, 1, &n, nullptr, 1, 8, CUDA_C_32F, nullptr, 1, 8,
                              CUDA_C_32F, 1, &work, CUDA_C_32F) == CUFFT_SUCCESS,
          "XtMakePlanMany");
    void* scratch = nullptr;
    CHECK(cudaMalloc(&scratch, work > 0 ? work : 16) == cudaSuccess, "scratch");
    void* areas[1] = {scratch};
    CHECK(cufftXtSetWorkArea(plan, areas) == CUFFT_SUCCESS, "XtSetWorkArea");
    int gpus[2] = {0, 0};
    CHECK(cufftXtSetGPUs(plan, 1, gpus) == CUFFT_SUCCESS, "XtSetGPUs one device");
    int other[1] = {1};
    CHECK(cufftXtSetGPUs(plan, 1, other) == CUFFT_INVALID_DEVICE, "device 1 accepted");
    CHECK(cufftXtSetGPUs(plan, 2, gpus) == CUFFT_NOT_SUPPORTED, "two GPUs accepted");

    cufftComplex* data = nullptr;
    CHECK(cudaMallocManaged(&data, 8 * sizeof(cufftComplex)) == cudaSuccess, "data");
    for (int i = 0; i < 8; ++i) data[i] = make_cuFloatComplex(static_cast<float>(i), 0.0f);
    CHECK(cufftXtExec(plan, data, data, CUFFT_FORWARD) == CUFFT_SUCCESS, "XtExec");
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "sync");
    // DFT of 0..7: X[0] = 28, X[k] = -4 + 4i*cot(pi k / 8) for k > 0.
    CHECK(std::fabs(data[0].x - 28.0f) < 1e-4f && std::fabs(data[0].y) < 1e-4f,
          "X[0] = %f%+fi", data[0].x, data[0].y);
    for (int k = 1; k < 8; ++k) {
        const float im = 4.0f / std::tan(3.14159265358979f * k / 8.0f);
        CHECK(std::fabs(data[k].x + 4.0f) < 1e-3f && std::fabs(data[k].y - im) < 1e-3f,
              "X[%d] = %f%+fi, want -4%+fi", k, data[k].x, data[k].y, im);
    }

    cudaLibXtDesc desc{};
    CHECK(cufftXtMemcpy(plan, &desc, data, CUFFT_COPY_HOST_TO_DEVICE) == CUFFT_NOT_SUPPORTED,
          "XtMemcpy");
    CHECK(cufftXtExecDescriptorC2C(plan, &desc, &desc, CUFFT_FORWARD) == CUFFT_NOT_SUPPORTED,
          "ExecDescriptorC2C");
    CHECK(cufftXtExecDescriptorZ2Z(plan, &desc, &desc, CUFFT_FORWARD) == CUFFT_NOT_SUPPORTED,
          "ExecDescriptorZ2Z");
    CHECK(cufftXtSetCallback(plan, nullptr, CUFFT_CB_LD_COMPLEX, nullptr) ==
              CUFFT_NOT_SUPPORTED,
          "XtSetCallback");
    CHECK(cufftDestroy(plan) == CUFFT_SUCCESS, "destroy");
    CHECK(cufftXtExec(plan, data, data, CUFFT_FORWARD) == CUFFT_INVALID_PLAN, "stale plan");
    CHECK(cufftSetAutoAllocation(plan, 1) == CUFFT_INVALID_PLAN, "stale plan auto-alloc");
    CHECK(cudaFree(data) == cudaSuccess && cudaFree(scratch) == cudaSuccess, "free");
    return true;
}

static bool test_nccl_218_surface() {
    static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES, "NCCL ids are 128 bytes");
    int version = 0;
    CHECK(ncclGetVersion(&version) == ncclSuccess && version == NCCL_VERSION_CODE,
          "ncclGetVersion %d vs NCCL_VERSION_CODE %d", version, NCCL_VERSION_CODE);

    ncclUniqueId id;
    CHECK(ncclGetUniqueId(&id) == ncclSuccess, "GetUniqueId");
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclComm_t comm = nullptr;
    CHECK(ncclCommInitRankConfig(&comm, 1, id, 0, &config) == ncclSuccess && comm != nullptr,
          "InitRankConfig");
    ncclResult_t async = ncclInternalError;
    CHECK(ncclCommGetAsyncError(comm, &async) == ncclSuccess && async == ncclSuccess,
          "GetAsyncError");

    ncclComm_t split = nullptr;
    CHECK(ncclCommSplit(comm, 0, 0, &split, &config) == ncclSuccess && split != nullptr,
          "Split");
    int count = 0, rank = -1;
    CHECK(ncclCommCount(split, &count) == ncclSuccess && count == 1, "split count");
    CHECK(ncclCommUserRank(split, &rank) == ncclSuccess && rank == 0, "split rank");
    ncclComm_t none = reinterpret_cast<ncclComm_t>(0x1);
    CHECK(ncclCommSplit(comm, NCCL_SPLIT_NOCOLOR, 0, &none, nullptr) == ncclSuccess &&
              none == nullptr,
          "NOCOLOR split must yield no communicator");

    float* buf = nullptr;
    CHECK(cudaMallocManaged(&buf, 4 * sizeof(float)) == cudaSuccess, "buf");
    for (int i = 0; i < 4; ++i) buf[i] = static_cast<float>(i + 1);
    CHECK(ncclBcast(buf, 4, ncclFloat, 0, comm, nullptr) == ncclSuccess, "Bcast");
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "sync");
    for (int i = 0; i < 4; ++i) CHECK(buf[i] == static_cast<float>(i + 1), "Bcast changed data");
    CHECK(ncclBcast(buf, 4, ncclFloat, 1, comm, nullptr) == ncclInvalidArgument, "root 1");
    CHECK(ncclBcast(nullptr, 4, ncclFloat, 0, comm, nullptr) == ncclInvalidArgument,
          "null buffer");

    CHECK(ncclCommDestroy(split) == ncclSuccess, "destroy split");
    CHECK(ncclCommDestroy(comm) == ncclSuccess, "destroy");
    CHECK(cudaFree(buf) == cudaSuccess, "free");
    return true;
}

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }
    const bool ok = test_pci_bus_id() && test_stream_priority() && test_mem_pool() &&
                    test_ipc_refuses() && test_array_async_copies() &&
                    test_link_single_image() && test_cufft_xt() && test_nccl_218_surface();
    if (!ok) return 1;
    std::printf("PASS: runtime CuPy surface\n");
    return 0;
}
