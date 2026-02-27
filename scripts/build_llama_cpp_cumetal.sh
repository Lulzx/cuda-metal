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

# ── create fake CUDA toolkit that CMake will accept ───────────────────────────
# cmake's CMakeCUDAFindToolkit.cmake probes nvcc in three ways:
#   1. nvcc --version  → parse "release X.Y" for version
#   2. nvcc -v __cmake_determine_cuda  → parse "#$ TOP=" and "#$ NVVMIR_LIBRARY_DIR="
#   3. Existence of ${toolkit_root}/nvvm/libdevice  → fallback LIBRARY_ROOT
# We handle all three.
FAKE_CUDA="${ROOT_DIR}/build/cumetal-cuda-toolkit"
mkdir -p \
    "${FAKE_CUDA}/bin" \
    "${FAKE_CUDA}/include" \
    "${FAKE_CUDA}/lib" \
    "${FAKE_CUDA}/lib64" \
    "${FAKE_CUDA}/nvvm/libdevice" \
    "${FAKE_CUDA}/lib/cmake/CUDAToolkit"

# Symlink CuMetal API headers into the fake CUDA include tree
for hdr in "${ROOT_DIR}/runtime/api/"*.h; do
    ln -sf "${hdr}" "${FAKE_CUDA}/include/$(basename "${hdr}")" 2>/dev/null || true
done

# Symlink CuMetal dylibs as CUDA runtime libraries (both lib/ and lib64/ for cmake compat)
for libdir in lib lib64; do
    ln -sf "${ROOT_DIR}/build/libcumetal.dylib" "${FAKE_CUDA}/${libdir}/libcudart.dylib"    2>/dev/null || true
    ln -sf "${ROOT_DIR}/build/libcumetal.dylib" "${FAKE_CUDA}/${libdir}/libcuda.dylib"      2>/dev/null || true
    for lib in cublas cufft curand; do
        src="${ROOT_DIR}/build/lib${lib}.dylib"
        [[ -f "${src}" ]] && ln -sf "${src}" "${FAKE_CUDA}/${libdir}/lib${lib}.dylib" 2>/dev/null || true
    done
done

# Version files — cmake parses these to detect CUDA 12.x
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
set(CUDAToolkit_LIBRARY_DIR  "\${_ctk_root}/lib")
set(CUDAToolkit_BIN_DIR      "\${_ctk_root}/bin")
set(CUDA_TOOLKIT_ROOT_DIR    "\${_ctk_root}")
set(CUDAToolkit_TARGET_DIR   "\${_ctk_root}")
if(NOT TARGET CUDA::cudart)
    add_library(CUDA::cudart SHARED IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "\${_ctk_root}/lib/libcudart.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "\${_ctk_root}/include"
    )
endif()
if(NOT TARGET CUDA::cublas)
    add_library(CUDA::cublas SHARED IMPORTED)
    set_target_properties(CUDA::cublas PROPERTIES
        IMPORTED_LOCATION "\${_ctk_root}/lib/libcublas.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "\${_ctk_root}/include"
    )
endif()
CMAKE

# nvcc shim — handles three cmake probes and delegates real compilation to clang++.
#
# cmake calls:
#   nvcc --version                        → print version banner
#   nvcc -v __cmake_determine_cuda        → print #$ TOP= and #$ NVVMIR_LIBRARY_DIR=
#   nvcc [compile flags] -o foo.o foo.cu  → compile via clang++ -x cuda
cat > "${FAKE_CUDA}/bin/nvcc" <<NVCC
#!/usr/bin/env bash
# CuMetal nvcc shim — delegates CUDA compilation to clang++

REAL_CLANG="${CLANG_BIN}"
CUMETAL_API="${ROOT_DIR}/runtime/api"
TOOLCHAIN="${ROOT_DIR}/scripts/cuda_toolchain"
FAKE_CUDA_ROOT="${FAKE_CUDA}"
export PATH="\${TOOLCHAIN}:\${PATH}"

# ── cmake version probe: nvcc --version ──────────────────────────────────────
if [[ "\$*" == "--version" || "\$*" == "-V" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2023 NVIDIA Corporation"
    echo "Cuda compilation tools, release 12.2, V12.2.140"
    exit 0
fi

# ── cmake toolkit probe: nvcc -v __cmake_determine_cuda ──────────────────────
# cmake parses stderr for "#$ TOP=" (toolkit root) and "#$ NVVMIR_LIBRARY_DIR="
# (library root, must end in nvvm/libdevice).
if [[ "\${1:-}" == "-v" ]]; then
    echo "#\$ TOP=\${FAKE_CUDA_ROOT}" >&2
    echo "#\$ NVVMIR_LIBRARY_DIR=\${FAKE_CUDA_ROOT}/nvvm/libdevice" >&2
    exit 0
fi

# ── real compilation: filter nvcc-only flags, delegate to clang++ ─────────────
ARGS=()
SKIP_NEXT=0
OPTIONS_FILE_NEXT=0
HAS_CUDA_SOURCE=0
COMPILE_ONLY=0
IS_CMAKE_PROBE=0
for arg in "\$@"; do
    if [[ \$OPTIONS_FILE_NEXT -eq 1 ]]; then
        ARGS+=("@\$arg")
        OPTIONS_FILE_NEXT=0
        continue
    fi
    if [[ \$SKIP_NEXT -eq 1 ]]; then SKIP_NEXT=0; continue; fi
    case "\$arg" in
        -c|-S|-E|-M|-MM|-MD|-MMD) COMPILE_ONLY=1 ;;
        *.cu)
            HAS_CUDA_SOURCE=1
            case "\$arg" in
                *CMakeCUDACompilerId.cu|*CMakeCUDACompilerABI.cu|*/CMakeScratch/*/*.cu)
                    IS_CMAKE_PROBE=1
                    ;;
            esac
            ;;
    esac
    case "\$arg" in
        # nvcc gencode flags — clang uses --cuda-gpu-arch instead
        -gencode)                              SKIP_NEXT=1; continue ;;
        --generate-code=*)                     continue ;;
        arch=compute_*|code=sm_*|code=lto_*)  continue ;;
        # nvcc forwarding wrappers / language selectors (clang is invoked directly)
        -forward-unknown-to-host-compiler|--forward-unknown-to-host-compiler) continue ;;
        -forward-unknown-to-host-linker|--forward-unknown-to-host-linker) continue ;;
        -x)                                    SKIP_NEXT=1; continue ;;
        -x=cu|-x=cpp|-x=c++)                   continue ;;
        # nvcc response-file syntax; clang understands @file directly
        --options-file|-optf)                  OPTIONS_FILE_NEXT=1; continue ;;
        --options-file=*|-optf=*)              ARGS+=("@\${arg#*=}"); continue ;;
        # nvcc language/feature toggles not needed for clang CUDA mode
        -extended-lambda|--extended-lambda)    continue ;;
        # nvcc compiler-identification / temp-file flags (cmake passes these)
        --keep)                               continue ;;
        --keep-dir)                           SKIP_NEXT=1; continue ;;
        # nvcc-only driver/linker flags
        --generate-dependencies-with-compile) continue ;;
        --dependency-output)                  SKIP_NEXT=1; continue ;;
        -dc|-dlink|-rdc=true|--relocatable-device-code=true) continue ;;
        # __cmake_determine_cuda is a placeholder file — ignore it
        __cmake_determine_cuda)               continue ;;
        # pass everything else through
        *) ARGS+=("\$arg") ;;
    esac
done

# Link-only invocations from CMake targets should be done as plain host links.
# Forcing clang CUDA mode on Darwin/LLVM 21 routes through clang-linker-wrapper,
# which currently mis-parses Apple's injected -lto_library linker flag.
if [[ \${HAS_CUDA_SOURCE} -eq 0 ]]; then
    exec "\${REAL_CLANG}" \\
        -Wno-unused-command-line-argument \\
        "\${ARGS[@]}"
fi

# CMake's CUDA compiler-ID/ABI probes are host-side programs and do not require
# device compilation. Compile+link them as C++ to avoid the CUDA offload linker.
if [[ \${IS_CMAKE_PROBE} -eq 1 && \${COMPILE_ONLY} -eq 0 ]]; then
    # CMake's nvcc parser expects a few nvcc-style "#$" metadata lines in the
    # verbose compiler output in order to infer host implicit link libraries.
    echo "#\$ PATH=\${PATH}" >&2
    # Use a library ordering that does not appear in clang's raw ld verbose line,
    # so CMake prefers the synthetic clang++ launcher line below.
    echo "#\$ LIBRARIES=-lSystem -lc++" >&2
    echo "#\$ INCLUDES=-I\${CUMETAL_API}" >&2
    echo "#\$ SYSTEM_INCLUDES=" >&2
    echo "\${REAL_CLANG} CMakeCUDACompilerId.o -lSystem -lc++" >&2
    exec "\${REAL_CLANG}" \\
        -x c++ \\
        -I"\${CUMETAL_API}" \\
        -D__CUDACC__=1 \\
        -D__NVCC__=1 \\
        -Wno-unused-command-line-argument \\
        "\${ARGS[@]}"
fi

exec "\${REAL_CLANG}" \\
    -x cuda \\
    --cuda-gpu-arch="${CUDA_ARCH}" \\
    -nocudainc -nocudalib \\
    -I"\${CUMETAL_API}" \\
    -include cuda_runtime.h \\
    -DCUMETAL_NO_DEVICE_PRINTF=1 \\
    -DCUDA_VERSION=11060 \\
    -DCUDART_VERSION=11060 \\
    -D__CUDACC__=1 \\
    -D__NVCC__=1 \\
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
    -DCMAKE_CUDA_COMPILER_LIBRARY_ROOT="${FAKE_CUDA}" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ROOT_DIR}/build -Wl,-rpath,${ROOT_DIR}/build" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ROOT_DIR}/build -Wl,-rpath,${ROOT_DIR}/build" \
    -DLLAMA_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DGGML_CUDA_NO_VMM=ON \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
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
