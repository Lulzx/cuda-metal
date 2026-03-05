#include "cublasLt.h"
#include "cublas_v2.h"

#include <Accelerate/Accelerate.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>

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
            std::fprintf(stderr, "[cublasLt] " fmt "\n", ##__VA_ARGS__);       \
    } while (0)

size_t element_size(cudaDataType_t dt) {
    switch (dt) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        case CUDA_R_32F:
        case CUDA_R_32I: return 4;
        case CUDA_R_64F: return 8;
        case CUDA_R_8I:
        case CUDA_R_8U: return 1;
        default: return 4;
    }
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
            std::memcpy(&matmulDesc->transa, buf, sizeof(cublasOperation_t));
            break;
        case CUBLASLT_MATMUL_DESC_TRANSB:
            if (sizeInBytes < sizeof(cublasOperation_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matmulDesc->transb, buf, sizeof(cublasOperation_t));
            break;
        case CUBLASLT_MATMUL_DESC_EPILOGUE:
            if (sizeInBytes < sizeof(cublasLtEpilogue_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matmulDesc->epilogue, buf, sizeof(cublasLtEpilogue_t));
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
            std::memcpy(&matLayout->batch_count, buf, sizeof(int32_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matLayout->strided_batch_offset, buf, sizeof(int64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_ROWS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matLayout->rows, buf, sizeof(uint64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_COLS:
            if (sizeInBytes < sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matLayout->cols, buf, sizeof(uint64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_LD:
            if (sizeInBytes < sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matLayout->ld, buf, sizeof(int64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_TYPE:
            if (sizeInBytes < sizeof(cudaDataType_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&matLayout->type, buf, sizeof(cudaDataType_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_ORDER:
            // Accept but ignore — we only support column-major
            break;
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

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t /*lightHandle*/,
                                               cublasLtMatmulDesc_t /*operationDesc*/,
                                               cublasLtMatrixLayout_t /*Adesc*/,
                                               cublasLtMatrixLayout_t /*Bdesc*/,
                                               cublasLtMatrixLayout_t /*Cdesc*/,
                                               cublasLtMatrixLayout_t /*Ddesc*/,
                                               cublasLtMatmulPreference_t /*preference*/,
                                               int requestedAlgoCount,
                                               cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                               int* returnAlgoCount) {
    if (!returnAlgoCount) return CUBLAS_STATUS_INVALID_VALUE;
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
                               const void* /*algo*/,
                               void* /*workspace*/, size_t /*workspaceSizeInBytes*/,
                               cudaStream_t /*stream*/) {
    if (!lightHandle || !computeDesc || !alpha || !A || !Adesc ||
        !B || !Bdesc || !beta || !C || !Cdesc || !D || !Ddesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

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
    if (batch_count < 1) batch_count = 1;

    LT_DEBUG("cublasLtMatmul m=%d n=%d k=%d batch=%d epilogue=%d",
             m, n, k, batch_count, (int)computeDesc->epilogue);

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
