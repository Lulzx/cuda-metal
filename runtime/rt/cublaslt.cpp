#include "cublasLt.h"
#include "cublas_v2.h"
#include "runtime_internal.h"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <vector>

struct cublasLtContext {
    cublasHandle_t cublas_handle = nullptr;
    bool owns_handle = false;
};

struct cublasLtMatmulDescOpaque {
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    const void* bias_pointer = nullptr;
};

struct cublasLtMatrixLayoutOpaque {
    cudaDataType_t type = CUDA_R_32F;
    uint64_t rows = 0;
    uint64_t cols = 0;
    int64_t ld = 0;
    int32_t batch_count = 1;
    int64_t strided_batch_offset = 0;
    int32_t order = 0; // CUBLASLT_ORDER_COL
};

struct cublasLtMatmulPreferenceOpaque {
    size_t max_workspace_bytes = 0;
};

namespace {

bool debug_cublaslt() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CUMETAL_DEBUG_CUBLASLT");
        v = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return v != 0;
}

#define LT_DEBUG(fmt, ...)                                                    \
    do {                                                                       \
        if (debug_cublaslt())                                                  \
            std::fprintf(stderr, "[cublasLt] " fmt "\n" __VA_OPT__(,) __VA_ARGS__); \
    } while (0)

bool valid_operation(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

bool valid_epilogue(cublasLtEpilogue_t epilogue) {
    return epilogue == CUBLASLT_EPILOGUE_DEFAULT ||
           epilogue == CUBLASLT_EPILOGUE_RELU ||
           epilogue == CUBLASLT_EPILOGUE_BIAS ||
           epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
           epilogue == CUBLASLT_EPILOGUE_GELU ||
           epilogue == CUBLASLT_EPILOGUE_GELU_BIAS;
}

std::size_t lt_element_size(cudaDataType_t type) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        case CUDA_R_64F: return sizeof(double);
        case CUDA_R_8I: return 1;
        default: return sizeof(float);
    }
}

float lt_bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::uint16_t lt_float_to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        // Round to nearest even on the dropped low half.
        bits += 0x7fffu + ((bits >> 16) & 1u);
    }
    return static_cast<std::uint16_t>(bits >> 16);
}

float lt_half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t out;
    if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            // Subnormal half: normalize.
            std::uint32_t m = mantissa;
            int e = -14;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            out = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

std::uint16_t lt_float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 112;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent >= 31) return sign | 0x7c00u; // Inf/overflow saturates to Inf.
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        std::uint32_t m = mantissa | 0x800000u;
        const int shift = 14 - exponent;
        std::uint16_t half = static_cast<std::uint16_t>(m >> shift);
        if ((m >> (shift - 1)) & 1u) ++half; // round to nearest
        return sign | half;
    }
    std::uint16_t half = static_cast<std::uint16_t>(
        sign | static_cast<std::uint16_t>(exponent << 10) |
        static_cast<std::uint16_t>(mantissa >> 13));
    if (mantissa & 0x1000u) ++half; // round to nearest
    return half;
}

float lt_read_float(const void* base, std::size_t index, cudaDataType_t type) {
    if (type == CUDA_R_16F)
        return lt_half_to_float(static_cast<const std::uint16_t*>(base)[index]);
    if (type == CUDA_R_16BF)
        return lt_bf16_to_float(static_cast<const std::uint16_t*>(base)[index]);
    if (type == CUDA_R_64F)
        return static_cast<float>(static_cast<const double*>(base)[index]);
    return static_cast<const float*>(base)[index];
}

void lt_write_float(void* base, std::size_t index, cudaDataType_t type, float value) {
    if (type == CUDA_R_16F) {
        static_cast<std::uint16_t*>(base)[index] = lt_float_to_half(value);
    } else if (type == CUDA_R_16BF) {
        static_cast<std::uint16_t*>(base)[index] = lt_float_to_bf16(value);
    } else if (type == CUDA_R_64F) {
        static_cast<double*>(base)[index] = value;
    } else {
        static_cast<float*>(base)[index] = value;
    }
}

bool valid_layout(const cublasLtMatrixLayoutOpaque* layout) {
    return layout && (layout->type == CUDA_R_32F || layout->type == CUDA_R_64F ||
                      layout->type == CUDA_R_16F || layout->type == CUDA_R_16BF ||
                      layout->type == CUDA_R_8I || layout->type == CUDA_R_32I) &&
           layout->order == 0 && layout->rows > 0 && layout->cols > 0 &&
           layout->rows <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
           layout->cols <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
           layout->ld >= static_cast<int64_t>(layout->rows) &&
           layout->ld <= std::numeric_limits<int>::max() &&
           layout->batch_count > 0 && layout->strided_batch_offset >= 0;
}

bool has_bias(cublasLtEpilogue_t epilogue) {
    return epilogue == CUBLASLT_EPILOGUE_BIAS ||
           epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
           epilogue == CUBLASLT_EPILOGUE_GELU_BIAS;
}

cublasStatus_t validate_matmul(cublasLtMatmulDesc_t computeDesc,
                               cublasLtMatrixLayout_t Adesc,
                               cublasLtMatrixLayout_t Bdesc,
                               cublasLtMatrixLayout_t Cdesc,
                               cublasLtMatrixLayout_t Ddesc) {
    if (!computeDesc || !valid_layout(Adesc) || !valid_layout(Bdesc) ||
        !valid_layout(Cdesc) || !valid_layout(Ddesc))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!valid_operation(computeDesc->transa) || !valid_operation(computeDesc->transb) ||
        !valid_epilogue(computeDesc->epilogue))
        return CUBLAS_STATUS_INVALID_VALUE;

    const cudaDataType_t type = Adesc->type;
    if (Bdesc->type != type || Cdesc->type != Ddesc->type)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (type == CUDA_R_16F) {
        // FP16 inputs with either FP16 output (16F compute) or FP32
        // output (32F compute); alpha/beta are FP32 in both cases.
        const bool half_out = Cdesc->type == CUDA_R_16F &&
                              computeDesc->compute_type == CUBLAS_COMPUTE_16F;
        const bool float_out = Cdesc->type == CUDA_R_32F &&
                               computeDesc->compute_type == CUBLAS_COMPUTE_32F;
        if ((!half_out && !float_out) || computeDesc->scale_type != CUDA_R_32F)
            return CUBLAS_STATUS_NOT_SUPPORTED;
    } else if (type == CUDA_R_16BF) {
        // BF16 inputs with BF16 or FP32 output under 32F compute.
        if ((Cdesc->type != CUDA_R_16BF && Cdesc->type != CUDA_R_32F) ||
            computeDesc->compute_type != CUBLAS_COMPUTE_32F ||
            computeDesc->scale_type != CUDA_R_32F)
            return CUBLAS_STATUS_NOT_SUPPORTED;
    } else if (type == CUDA_R_8I) {
        // INT8 inputs, INT32 accumulator/output and INT32 scales.
        if (Cdesc->type != CUDA_R_32I ||
            computeDesc->compute_type != CUBLAS_COMPUTE_32I ||
            computeDesc->scale_type != CUDA_R_32I ||
            computeDesc->epilogue != CUBLASLT_EPILOGUE_DEFAULT)
            return CUBLAS_STATUS_NOT_SUPPORTED;
    } else if (Cdesc->type != type ||
               (type == CUDA_R_32F &&
                (computeDesc->compute_type != CUBLAS_COMPUTE_32F ||
                 computeDesc->scale_type != CUDA_R_32F)) ||
               (type == CUDA_R_64F &&
                (computeDesc->compute_type != CUBLAS_COMPUTE_64F ||
                 computeDesc->scale_type != CUDA_R_64F)))
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (type == CUDA_R_64F && computeDesc->epilogue != CUBLASLT_EPILOGUE_DEFAULT)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (has_bias(computeDesc->epilogue) && !computeDesc->bias_pointer)
        return CUBLAS_STATUS_INVALID_VALUE;

    const uint64_t a_m = computeDesc->transa == CUBLAS_OP_N ? Adesc->rows : Adesc->cols;
    const uint64_t a_k = computeDesc->transa == CUBLAS_OP_N ? Adesc->cols : Adesc->rows;
    const uint64_t b_k = computeDesc->transb == CUBLAS_OP_N ? Bdesc->rows : Bdesc->cols;
    const uint64_t b_n = computeDesc->transb == CUBLAS_OP_N ? Bdesc->cols : Bdesc->rows;
    if (a_k != b_k || Ddesc->rows != a_m || Ddesc->cols != b_n ||
        Cdesc->rows != a_m || Cdesc->cols != b_n)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (Adesc->batch_count != Bdesc->batch_count ||
        Adesc->batch_count != Cdesc->batch_count ||
        Adesc->batch_count != Ddesc->batch_count)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    auto validate_batch_span = [](const cublasLtMatrixLayoutOpaque* layout) {
        const uint64_t span = static_cast<uint64_t>(layout->ld) * layout->cols;
        if (span > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            return CUBLAS_STATUS_INVALID_VALUE;
        if (layout->batch_count <= 1) return CUBLAS_STATUS_SUCCESS;
        const uint64_t offset = static_cast<uint64_t>(layout->strided_batch_offset);
        if (offset < span) return CUBLAS_STATUS_NOT_SUPPORTED;
        const uint64_t batches_after_first = static_cast<uint64_t>(layout->batch_count - 1);
        if (offset > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                         batches_after_first)
            return CUBLAS_STATUS_INVALID_VALUE;
        const uint64_t last_batch = offset * batches_after_first;
        if (span > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) -
                       last_batch)
            return CUBLAS_STATUS_INVALID_VALUE;
        return CUBLAS_STATUS_SUCCESS;
    };
    for (const auto* layout : {Adesc, Bdesc, Cdesc, Ddesc}) {
        cublasStatus_t batch_status = validate_batch_span(layout);
        if (batch_status != CUBLAS_STATUS_SUCCESS) return batch_status;
    }
    return CUBLAS_STATUS_SUCCESS;
}

bool tracked_span_is_valid(const void* pointer, const cublasLtMatrixLayoutOpaque* layout,
                           std::size_t element_size) {
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(pointer, &resolved)) {
        // This bounded CPU fallback intentionally accepts ordinary host buffers.
        return true;
    }
    const std::size_t matrix_elements =
        static_cast<std::size_t>(layout->ld) * static_cast<std::size_t>(layout->cols);
    const std::size_t batch_base = static_cast<std::size_t>(layout->batch_count - 1) *
                                   static_cast<std::size_t>(layout->strided_batch_offset);
    const std::size_t required_elements = batch_base + matrix_elements;
    return required_elements <= resolved.remaining_size / element_size;
}

bool tracked_vector_span_is_valid(const void* pointer, std::size_t elements,
                                  std::size_t element_size) {
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(pointer, &resolved)) return true;
    return elements <= resolved.remaining_size / element_size;
}

void apply_epilogue(float* D, int m, int n, int ldd,
                    cublasLtEpilogue_t epilogue, const void* bias) {
    bool has_bias = (epilogue == CUBLASLT_EPILOGUE_BIAS ||
                     epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
                     epilogue == CUBLASLT_EPILOGUE_GELU_BIAS);
    bool has_relu = (epilogue == CUBLASLT_EPILOGUE_RELU ||
                     epilogue == CUBLASLT_EPILOGUE_RELU_BIAS);
    bool has_gelu = (epilogue == CUBLASLT_EPILOGUE_GELU ||
                     epilogue == CUBLASLT_EPILOGUE_GELU_BIAS);

    if (!has_bias && !has_relu && !has_gelu) return;

    const float* b = static_cast<const float*>(bias);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float& val = D[j * ldd + i];
            if (has_bias && b) val += b[i];
            if (has_relu) val = val > 0.0f ? val : 0.0f;
            if (has_gelu) {
                // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                float x = val;
                float c = 0.7978845608f; // sqrt(2/π)
                val = x * 0.5f * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
            }
        }
    }
}

} // namespace

extern "C" {

cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle) {
    if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
    auto* ctx = new (std::nothrow) cublasLtContext;
    if (!ctx) return CUBLAS_STATUS_ALLOC_FAILED;

    cublasStatus_t st = cublasCreate(&ctx->cublas_handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        delete ctx;
        return st;
    }
    ctx->owns_handle = true;
    *lightHandle = ctx;
    LT_DEBUG("cublasLtCreate handle=%p", (void*)ctx);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
    LT_DEBUG("cublasLtDestroy handle=%p", (void*)lightHandle);
    if (lightHandle->owns_handle && lightHandle->cublas_handle) {
        cublasDestroy(lightHandle->cublas_handle);
    }
    delete lightHandle;
    return CUBLAS_STATUS_SUCCESS;
}

// --- Matmul descriptor ---

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                         cublasComputeType_t computeType,
                                         cudaDataType_t scaleType) {
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    if (!((computeType == CUBLAS_COMPUTE_32F && scaleType == CUDA_R_32F) ||
          (computeType == CUBLAS_COMPUTE_64F && scaleType == CUDA_R_64F) ||
          (computeType == CUBLAS_COMPUTE_16F && scaleType == CUDA_R_32F) ||
          (computeType == CUBLAS_COMPUTE_32I && scaleType == CUDA_R_32I)))
        return CUBLAS_STATUS_NOT_SUPPORTED;
    auto* d = new (std::nothrow) cublasLtMatmulDescOpaque;
    if (!d) return CUBLAS_STATUS_ALLOC_FAILED;
    d->compute_type = computeType;
    d->scale_type = scaleType;
    *matmulDesc = d;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    delete matmulDesc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatmulDescAttributes_t attr,
                                               const void* buf, size_t sizeInBytes) {
    if (!matmulDesc || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_TRANSA:
            if (sizeInBytes < sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                cublasOperation_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (!valid_operation(value)) return CUBLAS_STATUS_INVALID_VALUE;
                matmulDesc->transa = value;
            }
            break;
        case CUBLASLT_MATMUL_DESC_TRANSB:
            if (sizeInBytes < sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                cublasOperation_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (!valid_operation(value)) return CUBLAS_STATUS_INVALID_VALUE;
                matmulDesc->transb = value;
            }
            break;
        case CUBLASLT_MATMUL_DESC_EPILOGUE:
            if (sizeInBytes < sizeof(cublasLtEpilogue_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                cublasLtEpilogue_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (!valid_epilogue(value)) return CUBLAS_STATUS_INVALID_VALUE;
                matmulDesc->epilogue = value;
            }
            break;
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            if (sizeInBytes < sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matmulDesc->bias_pointer, buf, sizeof(void*));
            break;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatmulDescAttributes_t attr,
                                               void* buf, size_t sizeInBytes, size_t* sizeWritten) {
    if (!matmulDesc || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_TRANSA:
            if (sizeInBytes < sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matmulDesc->transa, sizeof(cublasOperation_t));
            if (sizeWritten) *sizeWritten = sizeof(cublasOperation_t);
            break;
        case CUBLASLT_MATMUL_DESC_TRANSB:
            if (sizeInBytes < sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matmulDesc->transb, sizeof(cublasOperation_t));
            if (sizeWritten) *sizeWritten = sizeof(cublasOperation_t);
            break;
        case CUBLASLT_MATMUL_DESC_EPILOGUE:
            if (sizeInBytes < sizeof(cublasLtEpilogue_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matmulDesc->epilogue, sizeof(cublasLtEpilogue_t));
            if (sizeWritten) *sizeWritten = sizeof(cublasLtEpilogue_t);
            break;
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            if (sizeInBytes < sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matmulDesc->bias_pointer, sizeof(void*));
            if (sizeWritten) *sizeWritten = sizeof(void*);
            break;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// --- Matrix layout ---

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout,
                                           cudaDataType_t type,
                                           uint64_t rows, uint64_t cols, int64_t ld) {
    if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
    if (type != CUDA_R_32F && type != CUDA_R_64F && type != CUDA_R_16F &&
        type != CUDA_R_16BF && type != CUDA_R_8I && type != CUDA_R_32I)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (rows == 0 || cols == 0 || rows > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        ld < static_cast<int64_t>(rows) || ld > std::numeric_limits<int>::max())
        return CUBLAS_STATUS_INVALID_VALUE;
    auto* l = new (std::nothrow) cublasLtMatrixLayoutOpaque;
    if (!l) return CUBLAS_STATUS_ALLOC_FAILED;
    l->type = type;
    l->rows = rows;
    l->cols = cols;
    l->ld = ld;
    *matLayout = l;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
    delete matLayout;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                 cublasLtMatrixLayoutAttribute_t attr,
                                                 const void* buf, size_t sizeInBytes) {
    if (!matLayout || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            if (sizeInBytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                int32_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value <= 0) return CUBLAS_STATUS_INVALID_VALUE;
                matLayout->batch_count = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                int64_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value < 0) return CUBLAS_STATUS_INVALID_VALUE;
                matLayout->strided_batch_offset = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_ROWS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                uint64_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                    matLayout->ld < static_cast<int64_t>(value))
                    return CUBLAS_STATUS_INVALID_VALUE;
                matLayout->rows = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_COLS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                uint64_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    return CUBLAS_STATUS_INVALID_VALUE;
                matLayout->cols = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_LD:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                int64_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value < static_cast<int64_t>(matLayout->rows) ||
                    value > std::numeric_limits<int>::max())
                    return CUBLAS_STATUS_INVALID_VALUE;
                matLayout->ld = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_TYPE:
            if (sizeInBytes < sizeof(cudaDataType_t)) return CUBLAS_STATUS_INVALID_VALUE;
            {
                cudaDataType_t value;
                std::memcpy(&value, buf, sizeof(value));
                if (value != CUDA_R_32F && value != CUDA_R_64F &&
                    value != CUDA_R_16F && value != CUDA_R_16BF)
                    return CUBLAS_STATUS_NOT_SUPPORTED;
                matLayout->type = value;
            }
            break;
        case CUBLASLT_MATRIX_LAYOUT_ORDER: {
            if (sizeInBytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            int32_t value;
            std::memcpy(&value, buf, sizeof(value));
            if (value != 0) return CUBLAS_STATUS_NOT_SUPPORTED;
            matLayout->order = value;
            break;
        }
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                 cublasLtMatrixLayoutAttribute_t attr,
                                                 void* buf, size_t sizeInBytes,
                                                 size_t* sizeWritten) {
    if (!matLayout || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATRIX_LAYOUT_TYPE:
            if (sizeInBytes < sizeof(cudaDataType_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->type, sizeof(cudaDataType_t));
            if (sizeWritten) *sizeWritten = sizeof(cudaDataType_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_ROWS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->rows, sizeof(uint64_t));
            if (sizeWritten) *sizeWritten = sizeof(uint64_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_COLS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->cols, sizeof(uint64_t));
            if (sizeWritten) *sizeWritten = sizeof(uint64_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_LD:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->ld, sizeof(int64_t));
            if (sizeWritten) *sizeWritten = sizeof(int64_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            if (sizeInBytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->batch_count, sizeof(int32_t));
            if (sizeWritten) *sizeWritten = sizeof(int32_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->strided_batch_offset, sizeof(int64_t));
            if (sizeWritten) *sizeWritten = sizeof(int64_t);
            break;
        case CUBLASLT_MATRIX_LAYOUT_ORDER: {
            if (sizeInBytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(buf, &matLayout->order, sizeof(int32_t));
            if (sizeWritten) *sizeWritten = sizeof(int32_t);
            break;
        }
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// --- Matmul preference ---

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    auto* p = new (std::nothrow) cublasLtMatmulPreferenceOpaque;
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    *pref = p;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    delete pref;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                      cublasLtMatmulPreferenceAttributes_t attr,
                                                      const void* buf, size_t sizeInBytes) {
    if (!pref || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) {
        if (sizeInBytes < sizeof(size_t)) return CUBLAS_STATUS_INVALID_VALUE;
        std::memcpy(&pref->max_workspace_bytes, buf, sizeof(size_t));
        return CUBLAS_STATUS_SUCCESS;
    }
    return CUBLAS_STATUS_INVALID_VALUE;
}

// --- Algorithm selection ---

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                               cublasLtMatmulDesc_t operationDesc,
                                               cublasLtMatrixLayout_t Adesc,
                                               cublasLtMatrixLayout_t Bdesc,
                                               cublasLtMatrixLayout_t Cdesc,
                                               cublasLtMatrixLayout_t Ddesc,
                                               cublasLtMatmulPreference_t preference,
                                               int requestedAlgoCount,
                                               cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                               int* returnAlgoCount) {
    if (!lightHandle || !preference || !returnAlgoCount || requestedAlgoCount < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    cublasStatus_t validation = validate_matmul(operationDesc, Adesc, Bdesc, Cdesc, Ddesc);
    if (validation != CUBLAS_STATUS_SUCCESS) {
        *returnAlgoCount = 0;
        return validation;
    }
    // We always return exactly one algorithm (Accelerate BLAS)
    if (requestedAlgoCount < 1 || !heuristicResultsArray) {
        *returnAlgoCount = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    std::memset(&heuristicResultsArray[0], 0, sizeof(cublasLtMatmulHeuristicResult_t));
    heuristicResultsArray[0].algo_id = 0;
    heuristicResultsArray[0].workspaceSize = 0;
    heuristicResultsArray[0].wavesCount = 1.0f;
    heuristicResultsArray[0].state = 0; // CUBLAS_STATUS_SUCCESS
    *returnAlgoCount = 1;
    return CUBLAS_STATUS_SUCCESS;
}

// --- Matmul execution ---

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                               cublasLtMatmulDesc_t computeDesc,
                               const void* alpha,
                               const void* A, cublasLtMatrixLayout_t Adesc,
                               const void* B, cublasLtMatrixLayout_t Bdesc,
                               const void* beta,
                               const void* C, cublasLtMatrixLayout_t Cdesc,
                               void* D, cublasLtMatrixLayout_t Ddesc,
                               const void* algo,
                               void* /*workspace*/, size_t /*workspaceSizeInBytes*/,
                               cudaStream_t stream) {
    if (!lightHandle || !computeDesc || !alpha || !A || !Adesc ||
        !B || !Bdesc || !beta || !C || !Cdesc || !D || !Ddesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (algo) return CUBLAS_STATUS_NOT_SUPPORTED;
    cublasStatus_t validation = validate_matmul(computeDesc, Adesc, Bdesc, Cdesc, Ddesc);
    if (validation != CUBLAS_STATUS_SUCCESS) return validation;
    if (!tracked_span_is_valid(A, Adesc, lt_element_size(Adesc->type)) ||
        !tracked_span_is_valid(B, Bdesc, lt_element_size(Bdesc->type)) ||
        !tracked_span_is_valid(C, Cdesc, lt_element_size(Cdesc->type)) ||
        !tracked_span_is_valid(D, Ddesc, lt_element_size(Ddesc->type)) ||
        (has_bias(computeDesc->epilogue) &&
         !tracked_vector_span_is_valid(computeDesc->bias_pointer,
                                       static_cast<std::size_t>(Ddesc->rows),
                                       lt_element_size(Ddesc->type)))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // Fast path: a plain FP32 GEMM (no epilogue) runs on the GPU through
    // MPSMatrixMultiplication, mirroring cublasSgemm, instead of falling
    // back to a host cblas GEMM over unified-memory pointers.
    if (computeDesc->compute_type == CUBLAS_COMPUTE_32F &&
        Adesc->type == CUDA_R_32F &&
        computeDesc->epilogue == CUBLASLT_EPILOGUE_DEFAULT) {
        cumetal::rt::AllocationTable::ResolvedAllocation a_res, b_res, c_res, d_res;
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        if (cumetal::rt::resolve_allocation_for_pointer(A, &a_res) &&
            cumetal::rt::resolve_allocation_for_pointer(B, &b_res) &&
            cumetal::rt::resolve_allocation_for_pointer(C, &c_res) &&
            cumetal::rt::resolve_allocation_for_pointer(D, &d_res) &&
            cumetal::rt::resolve_backend_stream(stream, &backend_stream) ==
                cudaSuccess) {
            const int lt_m = static_cast<int>(Ddesc->rows);
            const int lt_n = static_cast<int>(Ddesc->cols);
            const int lt_k = computeDesc->transa == CUBLAS_OP_N
                                 ? static_cast<int>(Adesc->cols)
                                 : static_cast<int>(Adesc->rows);
            const bool same_c_d = c_res.buffer == d_res.buffer &&
                                  c_res.offset == d_res.offset;
            const bool packed = Cdesc->ld == static_cast<int64_t>(Cdesc->rows) &&
                                Ddesc->ld == static_cast<int64_t>(Ddesc->rows);
            if (cumetal::rt::synchronize_for_host_library(stream) == cudaSuccess) {
                bool ok = true;
                if (!same_c_d) {
                    // MPS accumulates into its result matrix, so a distinct
                    // D must start from C's contents.
                    if (!packed) {
                        ok = false;
                    } else {
                        for (int b = 0; ok && b < Adesc->batch_count; ++b) {
                            const float* src = static_cast<const float*>(C) +
                                               b * Cdesc->strided_batch_offset;
                            float* dst = static_cast<float*>(D) +
                                         b * Ddesc->strided_batch_offset;
                            ok = cudaMemcpyAsync(dst, src,
                                                 static_cast<std::size_t>(lt_m) *
                                                     static_cast<std::size_t>(lt_n) *
                                                     sizeof(float),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream) == cudaSuccess;
                        }
                        if (ok) {
                            ok = cumetal::rt::synchronize_for_host_library(stream) ==
                                 cudaSuccess;
                        }
                    }
                }
                if (ok) {
                    std::string error;
                    cudaError_t gemm_status;
                    if (Adesc->batch_count == 1) {
                        gemm_status = cumetal::metal_backend::gemm_f32(
                            computeDesc->transa != CUBLAS_OP_N,
                            computeDesc->transb != CUBLAS_OP_N, lt_m, lt_n, lt_k,
                            *static_cast<const float*>(alpha), a_res.buffer,
                            a_res.offset, static_cast<int>(Adesc->ld), b_res.buffer,
                            b_res.offset, static_cast<int>(Bdesc->ld),
                            *static_cast<const float*>(beta), d_res.buffer,
                            d_res.offset, static_cast<int>(Ddesc->ld),
                            backend_stream, &error);
                    } else {
                        gemm_status =
                            cumetal::metal_backend::gemm_strided_batched_f32(
                                computeDesc->transa != CUBLAS_OP_N,
                                computeDesc->transb != CUBLAS_OP_N, lt_m, lt_n,
                                lt_k, *static_cast<const float*>(alpha),
                                a_res.buffer, a_res.offset,
                                static_cast<int>(Adesc->ld),
                                static_cast<std::size_t>(
                                    Adesc->strided_batch_offset) *
                                    sizeof(float),
                                b_res.buffer, b_res.offset,
                                static_cast<int>(Bdesc->ld),
                                static_cast<std::size_t>(
                                    Bdesc->strided_batch_offset) *
                                    sizeof(float),
                                *static_cast<const float*>(beta), d_res.buffer,
                                d_res.offset, static_cast<int>(Ddesc->ld),
                                static_cast<std::size_t>(
                                    Ddesc->strided_batch_offset) *
                                    sizeof(float),
                                Adesc->batch_count, backend_stream, &error);
                    }
                    if (gemm_status == cudaSuccess) {
                        return CUBLAS_STATUS_SUCCESS;
                    }
                    LT_DEBUG("MPS LtMatmul failed (%d): %s; host fallback",
                             static_cast<int>(gemm_status), error.c_str());
                }
            }
        }
    }

    // Fast path: FP32 GEMM with a fused epilogue (single batch, device
    // buffers) via the MPSGraph backend; anything it refuses falls
    // through to the host path below.
    if (computeDesc->compute_type == CUBLAS_COMPUTE_32F &&
        Adesc->type == CUDA_R_32F &&
        computeDesc->epilogue != CUBLASLT_EPILOGUE_DEFAULT &&
        Adesc->batch_count == 1) {
        cumetal::metal_backend::GemmEpilogue epi =
            cumetal::metal_backend::GemmEpilogue::kNone;
        switch (computeDesc->epilogue) {
            case CUBLASLT_EPILOGUE_RELU: epi = cumetal::metal_backend::GemmEpilogue::kRelu; break;
            case CUBLASLT_EPILOGUE_BIAS: epi = cumetal::metal_backend::GemmEpilogue::kBias; break;
            case CUBLASLT_EPILOGUE_RELU_BIAS: epi = cumetal::metal_backend::GemmEpilogue::kReluBias; break;
            case CUBLASLT_EPILOGUE_GELU: epi = cumetal::metal_backend::GemmEpilogue::kGelu; break;
            case CUBLASLT_EPILOGUE_GELU_BIAS: epi = cumetal::metal_backend::GemmEpilogue::kGeluBias; break;
            default: break;
        }
        cumetal::rt::AllocationTable::ResolvedAllocation a_res, b_res, c_res, d_res, bias_res;
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        const bool bias_ok = !has_bias(computeDesc->epilogue) ||
                             cumetal::rt::resolve_allocation_for_pointer(
                                 computeDesc->bias_pointer, &bias_res);
        if (epi != cumetal::metal_backend::GemmEpilogue::kNone && bias_ok &&
            cumetal::rt::resolve_allocation_for_pointer(A, &a_res) &&
            cumetal::rt::resolve_allocation_for_pointer(B, &b_res) &&
            cumetal::rt::resolve_allocation_for_pointer(C, &c_res) &&
            cumetal::rt::resolve_allocation_for_pointer(D, &d_res) &&
            cumetal::rt::resolve_backend_stream(stream, &backend_stream) ==
                cudaSuccess &&
            cumetal::rt::synchronize_for_host_library(stream) == cudaSuccess) {
            const int lt_m = static_cast<int>(Ddesc->rows);
            const int lt_n = static_cast<int>(Ddesc->cols);
            const int lt_k = computeDesc->transa == CUBLAS_OP_N
                                 ? static_cast<int>(Adesc->cols)
                                 : static_cast<int>(Adesc->rows);
            LT_DEBUG("MPSGraph LtMatmul epilogue path epilogue=%d",
                     static_cast<int>(computeDesc->epilogue));
            std::string error;
            const cudaError_t gemm_status = cumetal::metal_backend::gemm_f32_epilogue(
                computeDesc->transa != CUBLAS_OP_N,
                computeDesc->transb != CUBLAS_OP_N, lt_m, lt_n, lt_k,
                *static_cast<const float*>(alpha), a_res.buffer, a_res.offset,
                static_cast<int>(Adesc->ld), b_res.buffer, b_res.offset,
                static_cast<int>(Bdesc->ld),
                *static_cast<const float*>(beta), c_res.buffer, c_res.offset,
                static_cast<int>(Cdesc->ld), d_res.buffer, d_res.offset,
                static_cast<int>(Ddesc->ld),
                has_bias(computeDesc->epilogue) ? bias_res.buffer
                                                : std::shared_ptr<cumetal::metal_backend::Buffer>{},
                bias_res.offset, epi, backend_stream, &error);
            if (gemm_status == cudaSuccess) {
                return CUBLAS_STATUS_SUCCESS;
            }
            LT_DEBUG("MPSGraph LtMatmul epilogue failed (%d): %s; host fallback",
                     static_cast<int>(gemm_status), error.c_str());
        }
    }

    // Fast path: plain FP16 GEMM (no epilogue, single batch) on MPS via
    // gemm_f16 (FP16 out) or gemm_f16_f32 (FP32 out). Batched or
    // epilogue FP16 falls through to the float-domain host path below.
    if (Adesc->type == CUDA_R_16F &&
        computeDesc->epilogue == CUBLASLT_EPILOGUE_DEFAULT &&
        Adesc->batch_count == 1) {
        cumetal::rt::AllocationTable::ResolvedAllocation a_res, b_res, c_res, d_res;
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        if (cumetal::rt::resolve_allocation_for_pointer(A, &a_res) &&
            cumetal::rt::resolve_allocation_for_pointer(B, &b_res) &&
            cumetal::rt::resolve_allocation_for_pointer(C, &c_res) &&
            cumetal::rt::resolve_allocation_for_pointer(D, &d_res) &&
            cumetal::rt::resolve_backend_stream(stream, &backend_stream) ==
                cudaSuccess) {
            const int lt_m = static_cast<int>(Ddesc->rows);
            const int lt_n = static_cast<int>(Ddesc->cols);
            const int lt_k = computeDesc->transa == CUBLAS_OP_N
                                 ? static_cast<int>(Adesc->cols)
                                 : static_cast<int>(Adesc->rows);
            const bool same_c_d = c_res.buffer == d_res.buffer &&
                                  c_res.offset == d_res.offset;
            const bool packed = Cdesc->ld == static_cast<int64_t>(Cdesc->rows) &&
                                Ddesc->ld == static_cast<int64_t>(Ddesc->rows);
            if (cumetal::rt::synchronize_for_host_library(stream) == cudaSuccess) {
                bool ok = true;
                if (!same_c_d) {
                    if (!packed) {
                        ok = false;
                    } else {
                        ok = cudaMemcpyAsync(
                                 D, C,
                                 static_cast<std::size_t>(lt_m) *
                                     static_cast<std::size_t>(lt_n) *
                                     lt_element_size(Ddesc->type),
                                 cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
                             cumetal::rt::synchronize_for_host_library(stream) ==
                                 cudaSuccess;
                    }
                }
                if (ok) {
                    std::string error;
                    const float alpha_f = *static_cast<const float*>(alpha);
                    const float beta_f = *static_cast<const float*>(beta);
                    cudaError_t gemm_status;
                    if (Ddesc->type == CUDA_R_16F) {
                        gemm_status = cumetal::metal_backend::gemm_f16(
                            computeDesc->transa != CUBLAS_OP_N,
                            computeDesc->transb != CUBLAS_OP_N, lt_m, lt_n, lt_k,
                            alpha_f, a_res.buffer, a_res.offset,
                            static_cast<int>(Adesc->ld), b_res.buffer, b_res.offset,
                            static_cast<int>(Bdesc->ld), beta_f, d_res.buffer,
                            d_res.offset, static_cast<int>(Ddesc->ld),
                            backend_stream, &error);
                    } else {
                        gemm_status = cumetal::metal_backend::gemm_f16_f32(
                            computeDesc->transa != CUBLAS_OP_N,
                            computeDesc->transb != CUBLAS_OP_N, lt_m, lt_n, lt_k,
                            alpha_f, a_res.buffer, a_res.offset,
                            static_cast<int>(Adesc->ld), b_res.buffer, b_res.offset,
                            static_cast<int>(Bdesc->ld), beta_f, d_res.buffer,
                            d_res.offset, static_cast<int>(Ddesc->ld),
                            backend_stream, &error);
                    }
                    if (gemm_status == cudaSuccess) {
                        return CUBLAS_STATUS_SUCCESS;
                    }
                    LT_DEBUG("MPS LtMatmul FP16 failed (%d): %s; host fallback",
                             static_cast<int>(gemm_status), error.c_str());
                }
            }
        }
    }

    if (cumetal::rt::synchronize_for_host_library(stream) != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;

    int m = static_cast<int>(Ddesc->rows);
    int n = static_cast<int>(Ddesc->cols);

    // k: depends on transA
    int k;
    if (computeDesc->transa == CUBLAS_OP_N) {
        k = static_cast<int>(Adesc->cols);
    } else {
        k = static_cast<int>(Adesc->rows);
    }

    int lda = static_cast<int>(Adesc->ld);
    int ldb = static_cast<int>(Bdesc->ld);
    int ldc = static_cast<int>(Cdesc->ld);
    int ldd = static_cast<int>(Ddesc->ld);

    CBLAS_TRANSPOSE transA = (computeDesc->transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB = (computeDesc->transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;

    int batch_count = Adesc->batch_count;

    LT_DEBUG("cublasLtMatmul m=%d n=%d k=%d batch=%d epilogue=%d",
             m, n, k, batch_count, (int)computeDesc->epilogue);

    // FP16/BF16 host fallback: decode to float, run sgemm, apply any
    // epilogue in the float domain, and encode back. Correct for host
    // buffers, batched layouts, and epilogues the MPS path does not take.
    if (Adesc->type == CUDA_R_16F || Adesc->type == CUDA_R_16BF) {
        const float alpha_f = *static_cast<const float*>(alpha);
        const float beta_f = *static_cast<const float*>(beta);
        auto decode = [](const void* base, std::int64_t stride_elements,
                         int batch, const cublasLtMatrixLayoutOpaque* layout,
                         cudaDataType_t type) {
            const std::size_t count = static_cast<std::size_t>(layout->ld) *
                                      static_cast<std::size_t>(layout->cols);
            std::vector<float> out(count);
            const auto* src = static_cast<const std::uint8_t*>(base) +
                              static_cast<std::size_t>(batch * stride_elements) *
                                  lt_element_size(type);
            for (std::size_t i = 0; i < count; ++i)
                out[i] = lt_read_float(src, i, type);
            return out;
        };
        for (int b = 0; b < batch_count; ++b) {
            std::vector<float> af = decode(A, Adesc->strided_batch_offset, b,
                                           Adesc, Adesc->type);
            std::vector<float> bf = decode(B, Bdesc->strided_batch_offset, b,
                                           Bdesc, Bdesc->type);
            // Seed D from C in D's own leading dimension (ldc may differ).
            std::vector<float> df(static_cast<std::size_t>(ldd) *
                                  static_cast<std::size_t>(Ddesc->cols), 0.0f);
            {
                const auto* src = static_cast<const std::uint8_t*>(C) +
                                  static_cast<std::size_t>(
                                      b * Cdesc->strided_batch_offset) *
                                      lt_element_size(Cdesc->type);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        df[static_cast<std::size_t>(j * ldd + i)] =
                            lt_read_float(src,
                                          static_cast<std::size_t>(j * ldc + i),
                                          Cdesc->type);
            }
            cblas_sgemm(CblasColMajor, transA, transB, m, n, k, alpha_f,
                        af.data(), lda, bf.data(), ldb, beta_f, df.data(), ldd);
            if (computeDesc->epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
                std::vector<float> bias_f;
                const void* bias_arg = nullptr;
                if (has_bias(computeDesc->epilogue)) {
                    bias_f.resize(static_cast<std::size_t>(m));
                    for (int i = 0; i < m; ++i)
                        bias_f[static_cast<std::size_t>(i)] = lt_read_float(
                            computeDesc->bias_pointer,
                            static_cast<std::size_t>(i), Ddesc->type);
                    bias_arg = bias_f.data();
                }
                apply_epilogue(df.data(), m, n, ldd, computeDesc->epilogue,
                               bias_arg);
            }
            auto* dst = static_cast<std::uint8_t*>(D) +
                        static_cast<std::size_t>(b * Ddesc->strided_batch_offset) *
                            lt_element_size(Ddesc->type);
            const std::size_t count = static_cast<std::size_t>(ldd) *
                                      static_cast<std::size_t>(Ddesc->cols);
            for (std::size_t i = 0; i < count; ++i)
                lt_write_float(dst, i, Ddesc->type, df[i]);
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    // INT8 GEMM: exact integer accumulation (int64 internally, wrapped
    // to int32 on store, matching 32-bit accumulator semantics).
    if (Adesc->type == CUDA_R_8I) {
        const std::int32_t alpha_i = *static_cast<const std::int32_t*>(alpha);
        const std::int32_t beta_i = *static_cast<const std::int32_t*>(beta);
        const bool trans_a = computeDesc->transa != CUBLAS_OP_N;
        const bool trans_b = computeDesc->transb != CUBLAS_OP_N;
        for (int b = 0; b < batch_count; ++b) {
            const auto* ap = static_cast<const std::int8_t*>(A) +
                             b * Adesc->strided_batch_offset;
            const auto* bp = static_cast<const std::int8_t*>(B) +
                             b * Bdesc->strided_batch_offset;
            const auto* cp = static_cast<const std::int32_t*>(C) +
                             b * Cdesc->strided_batch_offset;
            auto* dp = static_cast<std::int32_t*>(D) +
                       b * Ddesc->strided_batch_offset;
            std::vector<std::int32_t> out(
                static_cast<std::size_t>(ldd) * static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    std::int64_t acc = 0;
                    for (int t = 0; t < k; ++t) {
                        const std::int64_t av = trans_a
                            ? ap[static_cast<std::size_t>(t) +
                                 static_cast<std::size_t>(i) * lda]
                            : ap[static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(t) * lda];
                        const std::int64_t bv = trans_b
                            ? bp[static_cast<std::size_t>(j) +
                                 static_cast<std::size_t>(t) * ldb]
                            : bp[static_cast<std::size_t>(t) +
                                 static_cast<std::size_t>(j) * ldb];
                        acc += av * bv;
                    }
                    const std::int64_t value =
                        static_cast<std::int64_t>(alpha_i) * acc +
                        static_cast<std::int64_t>(beta_i) *
                            cp[static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * ldc];
                    out[static_cast<std::size_t>(i) +
                        static_cast<std::size_t>(j) * ldd] =
                        static_cast<std::int32_t>(
                            static_cast<std::uint32_t>(value));
                }
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    dp[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(j) * ldd] =
                        out[static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(j) * ldd];
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    bool is_f64 = (computeDesc->compute_type == CUBLAS_COMPUTE_64F ||
                   Adesc->type == CUDA_R_64F);

    for (int b = 0; b < batch_count; ++b) {
        if (is_f64) {
            double a_val = *static_cast<const double*>(alpha);
            double b_val = *static_cast<const double*>(beta);

            const double* Ap = static_cast<const double*>(A) + b * Adesc->strided_batch_offset;
            const double* Bp = static_cast<const double*>(B) + b * Bdesc->strided_batch_offset;
            const double* Cp = static_cast<const double*>(C) + b * Cdesc->strided_batch_offset;
            double* Dp = static_cast<double*>(D) + b * Ddesc->strided_batch_offset;

            // D = alpha * op(A) * op(B) + beta * C
            // If D != C, copy C to D first so we can use D as both input and output
            if (Dp != Cp) {
                for (int j = 0; j < n; ++j)
                    std::memcpy(Dp + j * ldd, Cp + j * ldc, m * sizeof(double));
            }

            cblas_dgemm(CblasColMajor, transA, transB,
                        m, n, k, a_val,
                        Ap, lda, Bp, ldb,
                        b_val, Dp, ldd);
        } else {
            float a_val = *static_cast<const float*>(alpha);
            float b_val = *static_cast<const float*>(beta);

            const float* Ap = static_cast<const float*>(A) + b * Adesc->strided_batch_offset;
            const float* Bp = static_cast<const float*>(B) + b * Bdesc->strided_batch_offset;
            const float* Cp = static_cast<const float*>(C) + b * Cdesc->strided_batch_offset;
            float* Dp = static_cast<float*>(D) + b * Ddesc->strided_batch_offset;

            if (Dp != Cp) {
                for (int j = 0; j < n; ++j)
                    std::memcpy(Dp + j * ldd, Cp + j * ldc, m * sizeof(float));
            }

            cblas_sgemm(CblasColMajor, transA, transB,
                        m, n, k, a_val,
                        Ap, lda, Bp, ldb,
                        b_val, Dp, ldd);

            // Apply epilogue (bias, relu, gelu) — only for float path
            if (computeDesc->epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
                apply_epilogue(Dp, m, n, ldd, computeDesc->epilogue,
                               computeDesc->bias_pointer);
            }
        }
    }

    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
