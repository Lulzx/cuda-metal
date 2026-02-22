#!/usr/bin/env bash
# build_llama_cpp_cumetal.sh — Build llama.cpp (GGML CUDA backend) via CuMetal
#
# Usage:
#   bash scripts/build_llama_cpp_cumetal.sh [llama-cpp-dir]
#
# Environment overrides:
#   CUMETAL_LLAMA_DIR    path to llama.cpp checkout (default: ../llama.cpp)
#   CUMETAL_LLAMA_REPO   git remote to clone from (default: https://github.com/ggml-org/llama.cpp)
#   CUMETAL_LLAMA_TAG    git tag/branch to pin (default: latest main)
#   CUMETAL_CLANG        clang++ binary to use (default: auto-detect)
#   CUMETAL_CUDA_ARCH    CUDA arch string (default: sm_80)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── configuration ─────────────────────────────────────────────────────────────
LLAMA_DIR="${CUMETAL_LLAMA_DIR:-${1:-${ROOT_DIR}/../llama.cpp}}"
LLAMA_REPO="${CUMETAL_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_TAG="${CUMETAL_LLAMA_TAG:-}"        # empty = latest main
CUDA_ARCH="${CUMETAL_CUDA_ARCH:-sm_80}"
NCPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# ── find clang++ ──────────────────────────────────────────────────────────────
CLANG_BIN="${CUMETAL_CLANG:-}"
if [[ -z "${CLANG_BIN}" ]]; then
    for candidate in \
        /opt/homebrew/opt/llvm/bin/clang++ \
        /usr/local/opt/llvm/bin/clang++ \
        "$(command -v clang++ 2>/dev/null || true)"
    do
        [[ -x "${candidate}" ]] && { CLANG_BIN="${candidate}"; break; }
    done
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "ERROR: clang++ not found; install with: brew install llvm" >&2
    exit 2
fi
echo "clang++: ${CLANG_BIN}"

# ── clone llama.cpp if needed ─────────────────────────────────────────────────
if [[ ! -d "${LLAMA_DIR}" ]]; then
    echo "Cloning llama.cpp → ${LLAMA_DIR} ..."
    if [[ -n "${LLAMA_TAG}" ]]; then
        git clone --depth 1 --branch "${LLAMA_TAG}" "${LLAMA_REPO}" "${LLAMA_DIR}"
    else
        git clone --depth 1 "${LLAMA_REPO}" "${LLAMA_DIR}"
    fi
fi
echo "llama.cpp source: ${LLAMA_DIR}"

# ── create fake CUDA toolkit that CMake's find_package(CUDAToolkit) will accept
# cmake looks for: <root>/bin/nvcc, <root>/include/, version.json or version.txt
FAKE_CUDA="${ROOT_DIR}/build/cumetal-cuda-toolkit"
mkdir -p "${FAKE_CUDA}/bin" "${FAKE_CUDA}/include" "${FAKE_CUDA}/lib64" \
         "${FAKE_CUDA}/lib/cmake/CUDAToolkit"

# Symlink all CuMetal API headers into the fake CUDA include tree
for hdr in "${ROOT_DIR}/runtime/api/"*.h; do
    ln -sf "${hdr}" "${FAKE_CUDA}/include/$(basename "${hdr}")" 2>/dev/null || true
done

# Symlink CuMetal dylibs as the CUDA runtime libraries
ln -sf "${ROOT_DIR}/build/libcumetal.dylib"  "${FAKE_CUDA}/lib64/libcudart.dylib"     2>/dev/null || true
ln -sf "${ROOT_DIR}/build/libcumetal.dylib"  "${FAKE_CUDA}/lib64/libcudart_static.a"  2>/dev/null || true
ln -sf "${ROOT_DIR}/build/libcumetal.dylib"  "${FAKE_CUDA}/lib64/libcuda.dylib"       2>/dev/null || true
for lib in cublas cufft curand; do
    src="${ROOT_DIR}/build/lib${lib}.dylib"
    [[ -f "${src}" ]] && ln -sf "${src}" "${FAKE_CUDA}/lib64/lib${lib}.dylib" 2>/dev/null || true
done

# Write version file — cmake parses this to detect CUDA 12.x
cat > "${FAKE_CUDA}/version.json" <<'JSON'
{
   "cuda" : { "version" : "12.2.0" }
}
JSON
echo "CUDA Version 12.2.0" > "${FAKE_CUDA}/version.txt"

# CUDAToolkit cmake config — satisfies find_package(CUDAToolkit REQUIRED)
cat > "${FAKE_CUDA}/lib/cmake/CUDAToolkit/CUDAToolkitConfig.cmake" <<CMAKE
set(CUDAToolkit_VERSION "12.2.0")
set(CUDAToolkit_VERSION_MAJOR 12)
set(CUDAToolkit_VERSION_MINOR 2)
set(CUDAToolkit_VERSION_PATCH 0)
get_filename_component(_ctk_root "\${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(CUDAToolkit_INCLUDE_DIRS "\${_ctk_root}/include")
set(CUDAToolkit_LIBRARY_DIR  "\${_ctk_root}/lib64")
set(CUDAToolkit_BIN_DIR      "\${_ctk_root}/bin")
set(CUDA_TOOLKIT_ROOT_DIR    "\${_ctk_root}")
set(CUDAToolkit_TARGET_DIR   "\${_ctk_root}")
if(NOT TARGET CUDA::cudart)
    add_library(CUDA::cudart SHARED IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "\${_ctk_root}/lib64/libcudart.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "\${_ctk_root}/include"
    )
endif()
if(NOT TARGET CUDA::cublas)
    add_library(CUDA::cublas SHARED IMPORTED)
    set_target_properties(CUDA::cublas PROPERTIES
        IMPORTED_LOCATION "\${_ctk_root}/lib64/libcublas.dylib"
    )
endif()
CMAKE

# nvcc shim — cmake interrogates "nvcc --version" and then invokes nvcc for .cu files.
# We intercept both and delegate real compilation to clang++ -x cuda.
cat > "${FAKE_CUDA}/bin/nvcc" <<NVCC
#!/usr/bin/env bash
# CuMetal nvcc shim — delegates CUDA compilation to clang++
REAL_CLANG="${CLANG_BIN}"
CUMETAL_API="${ROOT_DIR}/runtime/api"
TOOLCHAIN="${ROOT_DIR}/scripts/cuda_toolchain"
export PATH="\${TOOLCHAIN}:\${PATH}"

# Handle cmake's version probe
if [[ "\$*" == "--version" || "\$*" == "-V" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2023 NVIDIA Corporation"
    echo "Cuda compilation tools, release 12.2, V12.2.140"
    exit 0
fi

# Filter out nvcc-only flags that clang does not accept and build translated args
ARGS=()
SKIP_NEXT=0
for arg in "\$@"; do
    if [[ \$SKIP_NEXT -eq 1 ]]; then SKIP_NEXT=0; continue; fi
    case "\$arg" in
        # nvcc code-generation flags → translate to clang --cuda-gpu-arch
        -gencode)       SKIP_NEXT=1; continue ;;
        arch=compute_*) continue ;;
        code=sm_*)      continue ;;
        # nvcc-only output flags we don't need
        --generate-dependencies-with-compile) continue ;;
        --dependency-output) SKIP_NEXT=1; continue ;;
        -dc|-dlink)     continue ;;
        # pass everything else through
        *) ARGS+=("\$arg") ;;
    esac
done

exec "\${REAL_CLANG}" \\
    -x cuda \\
    --cuda-gpu-arch="${CUDA_ARCH}" \\
    -nocudainc -nocudalib \\
    -I"\${CUMETAL_API}" \\
    -Wno-pass-failed \\
    -Wno-unknown-cuda-version \\
    -Wno-unused-command-line-argument \\
    "\${ARGS[@]}"
NVCC
chmod +x "${FAKE_CUDA}/bin/nvcc"

echo "Fake CUDA toolkit ready: ${FAKE_CUDA}"

# ── configure llama.cpp ───────────────────────────────────────────────────────
LLAMA_BUILD="${LLAMA_DIR}/build-cumetal"
mkdir -p "${LLAMA_BUILD}"

echo "Configuring llama.cpp ..."
PATH="${FAKE_CUDA}/bin:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_METAL=OFF \
    -DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="${FAKE_CUDA}" \
    -DCUDAToolkit_ROOT="${FAKE_CUDA}" \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ROOT_DIR}/build -Wl,-rpath,${ROOT_DIR}/build" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ROOT_DIR}/build -Wl,-rpath,${ROOT_DIR}/build" \
    -DLLAMA_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF \
    2>&1

# ── build llama-cli ───────────────────────────────────────────────────────────
echo "Building llama-cli (j=${NCPUS}) ..."
PATH="${FAKE_CUDA}/bin:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}" \
cmake --build "${LLAMA_BUILD}" --target llama-cli -j"${NCPUS}" 2>&1

LLAMA_CLI="${LLAMA_BUILD}/bin/llama-cli"
if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "ERROR: build succeeded but ${LLAMA_CLI} not found" >&2
    exit 1
fi

echo ""
echo "SUCCESS: built ${LLAMA_CLI}"
echo "Run conformance test: bash tests/conformance/run_llama_cpp_cumetal.sh"
