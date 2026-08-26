#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>

static bool test_malloc_free_array() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    if (desc.x != 32 || desc.f != cudaChannelFormatKindFloat) {
        std::fprintf(stderr, "FAIL: cudaCreateChannelDesc returned wrong values\n");
        return false;
    }

    cudaArray_t arr = nullptr;
    cudaError_t err = cudaMallocArray(&arr, &desc, 64, 64, 0);
    if (err != cudaSuccess || arr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMallocArray returned %d\n", err);
        return false;
    }

    err = cudaFreeArray(arr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeArray returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_texture_object_lifecycle() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t arr = nullptr;
    cudaMallocArray(&arr, &desc, 16, 16, 0);

    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceDesc::cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaResourceViewDesc viewDesc;
    std::memset(&viewDesc, 0, sizeof(viewDesc));
    viewDesc.width = 8;
    viewDesc.height = 7;
    viewDesc.firstLayer = 2;
    viewDesc.lastLayer = 5;

    cudaTextureObject_t texObj = 0;
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, &viewDesc);
    if (err != cudaSuccess || texObj == 0) {
        std::fprintf(stderr, "FAIL: cudaCreateTextureObject returned %d\n", err);
        return false;
    }

    cudaResourceDesc queriedResDesc;
    cudaTextureDesc queriedTexDesc;
    cudaResourceViewDesc queriedViewDesc;
    std::memset(&queriedResDesc, 0, sizeof(queriedResDesc));
    std::memset(&queriedTexDesc, 0, sizeof(queriedTexDesc));
    std::memset(&queriedViewDesc, 0, sizeof(queriedViewDesc));
    if (cudaGetTextureObjectResourceDesc(&queriedResDesc, texObj) != cudaSuccess ||
        cudaGetTextureObjectTextureDesc(&queriedTexDesc, texObj) != cudaSuccess ||
        cudaGetTextureObjectResourceViewDesc(&queriedViewDesc, texObj) != cudaSuccess ||
        queriedResDesc.resType != cudaResourceDesc::cudaResourceTypeArray ||
        queriedResDesc.res.array.array != arr ||
        queriedTexDesc.addressMode[0] != cudaAddressModeClamp ||
        queriedTexDesc.filterMode != cudaFilterModePoint ||
        queriedTexDesc.normalizedCoords != 1 ||
        queriedViewDesc.width != 8 || queriedViewDesc.height != 7 ||
        queriedViewDesc.firstLayer != 2 || queriedViewDesc.lastLayer != 5) {
        std::fprintf(stderr, "FAIL: texture descriptor round trip\n");
        return false;
    }
    if (cudaGetTextureObjectResourceDesc(nullptr, texObj) != cudaErrorInvalidValue ||
        cudaCreateTextureObject(nullptr, &resDesc, &texDesc, nullptr) !=
            cudaErrorInvalidValue ||
        cudaCreateTextureObject(&texObj, &resDesc, nullptr, nullptr) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: texture descriptor negative path\n");
        return false;
    }

    err = cudaDestroyTextureObject(texObj);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDestroyTextureObject returned %d\n", err);
        return false;
    }
    if (cudaDestroyTextureObject(texObj) != cudaErrorInvalidResourceHandle ||
        cudaGetTextureObjectTextureDesc(&queriedTexDesc, texObj) !=
            cudaErrorInvalidResourceHandle) {
        std::fprintf(stderr, "FAIL: stale texture handle accepted\n");
        return false;
    }

    cudaFreeArray(arr);
    return true;
}

static bool test_surface_object_lifecycle() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t arr = nullptr;
    cudaMallocArray(&arr, &desc, 32, 32, 0);

    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceDesc::cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaSurfaceObject_t surfObj = 0;
    cudaError_t err = cudaCreateSurfaceObject(&surfObj, &resDesc);
    if (err != cudaSuccess || surfObj == 0) {
        std::fprintf(stderr, "FAIL: cudaCreateSurfaceObject returned %d\n", err);
        return false;
    }

    cudaResourceDesc queriedResDesc;
    std::memset(&queriedResDesc, 0, sizeof(queriedResDesc));
    if (cudaGetSurfaceObjectResourceDesc(&queriedResDesc, surfObj) != cudaSuccess ||
        queriedResDesc.resType != cudaResourceDesc::cudaResourceTypeArray ||
        queriedResDesc.res.array.array != arr ||
        cudaGetSurfaceObjectResourceDesc(nullptr, surfObj) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: surface resource descriptor round trip\n");
        return false;
    }

    err = cudaDestroySurfaceObject(surfObj);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDestroySurfaceObject returned %d\n", err);
        return false;
    }
    if (cudaDestroySurfaceObject(surfObj) != cudaErrorInvalidResourceHandle ||
        cudaGetSurfaceObjectResourceDesc(&queriedResDesc, surfObj) !=
            cudaErrorInvalidResourceHandle) {
        std::fprintf(stderr, "FAIL: stale surface handle accepted\n");
        return false;
    }

    cudaFreeArray(arr);
    return true;
}

static bool test_memcpy_to_from_array() {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t arr = nullptr;
    const size_t w = 4, h = 4;
    cudaMallocArray(&arr, &desc, w, h, 0);

    float src[16];
    for (int i = 0; i < 16; ++i) src[i] = static_cast<float>(i);

    cudaError_t err = cudaMemcpyToArray(arr, 0, 0, src, sizeof(src), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyToArray returned %d\n", err);
        return false;
    }

    float dst[16] = {};
    err = cudaMemcpyFromArray(dst, arr, 0, 0, sizeof(dst), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpyFromArray returned %d\n", err);
        return false;
    }

    for (int i = 0; i < 16; ++i) {
        if (dst[i] != src[i]) {
            std::fprintf(stderr, "FAIL: memcpy roundtrip mismatch at [%d]: %f != %f\n",
                         i, dst[i], src[i]);
            return false;
        }
    }

    cudaFreeArray(arr);
    return true;
}

static bool test_channel_format_desc_variants() {
    // RGBA 8-bit unsigned
    cudaChannelFormatDesc d1 = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    if (d1.x != 8 || d1.y != 8 || d1.z != 8 || d1.w != 8 ||
        d1.f != cudaChannelFormatKindUnsigned) {
        std::fprintf(stderr, "FAIL: RGBA8 channel desc\n");
        return false;
    }

    // Single 16-bit signed
    cudaChannelFormatDesc d2 = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned);
    if (d2.x != 16 || d2.y != 0 || d2.f != cudaChannelFormatKindSigned) {
        std::fprintf(stderr, "FAIL: R16S channel desc\n");
        return false;
    }
    return true;
}

int main() {
    if (!test_malloc_free_array()) return 1;
    if (!test_texture_object_lifecycle()) return 1;
    if (!test_surface_object_lifecycle()) return 1;
    if (!test_memcpy_to_from_array()) return 1;
    if (!test_channel_format_desc_variants()) return 1;

    std::printf("PASS: Texture/Surface API tests\n");
    return 0;
}
