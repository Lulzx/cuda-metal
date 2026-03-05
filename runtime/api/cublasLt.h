#pragma once

#include "cublas_v2.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque* cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque* cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulPreferenceOpaque* cublasLtMatmulPreference_t;


typedef enum cublasLtMatmulDescAttributes_t {
    CUBLASLT_MATMUL_DESC_TRANSA = 0,
    CUBLASLT_MATMUL_DESC_TRANSB = 1,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 2,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 3,
} cublasLtMatmulDescAttributes_t;

typedef enum cublasLtMatrixLayoutAttribute_t {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,
} cublasLtMatrixLayoutAttribute_t;

typedef enum cublasLtMatmulPreferenceAttributes_t {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} cublasLtMatmulPreferenceAttributes_t;

typedef enum cublasLtEpilogue_t {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_RELU = 2,
    CUBLASLT_EPILOGUE_BIAS = 4,
    CUBLASLT_EPILOGUE_RELU_BIAS = 6,
    CUBLASLT_EPILOGUE_GELU = 32,
    CUBLASLT_EPILOGUE_GELU_BIAS = 36,
} cublasLtEpilogue_t;

typedef struct cublasLtMatmulHeuristicResult_t {
    int algo_id;
    size_t workspaceSize;
    float wavesCount;
    int state;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

// Handle management
cublasStatus_t cublasLtCreate(cublasLtHandle_t* lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);

// Matmul descriptor
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                         cublasComputeType_t computeType,
                                         cudaDataType_t scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatmulDescAttributes_t attr,
                                               const void* buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatmulDescAttributes_t attr,
                                               void* buf, size_t sizeInBytes, size_t* sizeWritten);

// Matrix layout
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout,
                                           cudaDataType_t type,
                                           uint64_t rows, uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                 cublasLtMatrixLayoutAttribute_t attr,
                                                 const void* buf, size_t sizeInBytes);

// Matmul preference
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                      cublasLtMatmulPreferenceAttributes_t attr,
                                                      const void* buf, size_t sizeInBytes);

// Algorithm selection
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                               cublasLtMatmulDesc_t operationDesc,
                                               cublasLtMatrixLayout_t Adesc,
                                               cublasLtMatrixLayout_t Bdesc,
                                               cublasLtMatrixLayout_t Cdesc,
                                               cublasLtMatrixLayout_t Ddesc,
                                               cublasLtMatmulPreference_t preference,
                                               int requestedAlgoCount,
                                               cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                               int* returnAlgoCount);

// Matmul execution
cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                               cublasLtMatmulDesc_t computeDesc,
                               const void* alpha,
                               const void* A, cublasLtMatrixLayout_t Adesc,
                               const void* B, cublasLtMatrixLayout_t Bdesc,
                               const void* beta,
                               const void* C, cublasLtMatrixLayout_t Cdesc,
                               void* D, cublasLtMatrixLayout_t Ddesc,
                               const void* algo,
                               void* workspace, size_t workspaceSizeInBytes,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
