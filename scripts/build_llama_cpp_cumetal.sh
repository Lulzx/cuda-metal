#!/usr/bin/env bash
# build_llama_cpp_cumetal.sh — Build llama.cpp (GGML CUDA backend) via CuMetal
#
# Usage:
#   bash scripts/build_llama_cpp_cumetal.sh [llama-cpp-dir]
#   bash scripts/build_llama_cpp_cumetal.sh --toolkit-only
#
# Environment overrides:
#   CUMETAL_LLAMA_DIR    path to llama.cpp checkout (default: ../llama.cpp)
#   CUMETAL_LLAMA_REPO   git remote to clone from (default: https://github.com/ggml-org/llama.cpp)
#   CUMETAL_LLAMA_TAG    git tag/branch to pin (default: latest main)
#   CUMETAL_CLANG        clang++ binary to use (default: auto-detect)
#   CUMETAL_CUDA_ARCH    CUDA arch string (default: sm_80)
#
# CuMetal does not currently lower llama.cpp's fused FlashAttention kernels.
# The build therefore disables GGML_CUDA_FA so llama.cpp's normal backend
# capability probe selects its supported non-fused attention graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/cumetal_cuda_flags.sh
source "${SCRIPT_DIR}/cumetal_cuda_flags.sh"

# ── configuration ─────────────────────────────────────────────────────────────
TOOLKIT_ONLY=0
if [[ "${1:-}" == "--toolkit-only" ]]; then
    TOOLKIT_ONLY=1
    shift
fi
LLAMA_DIR="${CUMETAL_LLAMA_DIR:-${1:-${ROOT_DIR}/../llama.cpp}}"
LLAMA_REPO="${CUMETAL_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_TAG="${CUMETAL_LLAMA_TAG:-}"        # empty = latest main
cumetal_cuda_device_flags
CUDA_ARCH="$(cumetal_cuda_arch)"
CUDA_DEVICE_FLAGS_STR="${CUMETAL_CUDA_DEVICE_FLAGS[*]}"
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

# ── nvcc -dlink: relocatable device-code link ────────────────────────────────
# CUDA_SEPARABLE_COMPILATION makes cmake run a device-link pass that fuses the
# per-TU device images into one object. CuMetal has no such image: each TU
# registers its own kernels with the runtime at load time, so there is nothing
# to fuse. Emit an empty object so the host link that follows still finds the
# file cmake told it to expect. Falling through to the normal path instead
# would attempt a real host link of the device objects against
# deviceLinkLibs.rsp, which lists no CUDA runtime and so fails on every
# __cudaRegister* symbol.
DLINK=0
DLINK_OUT=""
WANT_OUT=0
for arg in "\$@"; do
    if [[ \$WANT_OUT -eq 1 ]]; then DLINK_OUT="\$arg"; WANT_OUT=0; continue; fi
    case "\$arg" in
        -dlink|--device-link) DLINK=1 ;;
        -o)                   WANT_OUT=1 ;;
        -o*)                  DLINK_OUT="\${arg#-o}" ;;
    esac
done
if [[ \$DLINK -eq 1 ]]; then
    if [[ -z "\$DLINK_OUT" ]]; then
        echo "nvcc shim: -dlink without -o" >&2
        exit 1
    fi
    exec "\${REAL_CLANG}" -x c++ -c /dev/null -o "\$DLINK_OUT"
fi

# ── real compilation: filter nvcc-only flags, delegate to clang++ ─────────────
ARGS=()
SKIP_NEXT=0
OPTIONS_FILE_NEXT=0
XCOMPILER_NEXT=0
HAS_CUDA_SOURCE=0
COMPILE_ONLY=0
IS_CMAKE_PROBE=0
for arg in "\$@"; do
    if [[ \$XCOMPILER_NEXT -eq 1 ]]; then
        OLDIFS="\$IFS"; IFS=','
        for hostflag in \$arg; do ARGS+=("\$hostflag"); done
        IFS="\$OLDIFS"
        XCOMPILER_NEXT=0
        continue
    fi
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
        # nvcc -arch / --gpu-architecture: cmake emits these from
        # CMAKE_CUDA_ARCHITECTURES (including the literal "all"). clang selects
        # the arch via --cuda-gpu-arch, which this shim already appends.
        -arch|--gpu-architecture)              SKIP_NEXT=1; continue ;;
        -arch=*|--gpu-architecture=*)          continue ;;
        # nvcc host-compiler passthrough. cmake emits -Xcompiler=-fPIC for any
        # shared CUDA library target. The value is a comma-separated list of
        # host flags, which clang takes directly.
        -Xcompiler|--compiler-options)         XCOMPILER_NEXT=1; continue ;;
        -Xcompiler=*|--compiler-options=*)
            OLDIFS="\$IFS"; IFS=','
            for hostflag in \${arg#*=}; do ARGS+=("\$hostflag"); done
            IFS="\$OLDIFS"
            continue ;;
        # Flags for nvcc-internal sub-tools that clang has no equivalent for.
        -Xptxas|-Xcudafe|-Xnvlink|-Xarchive)   SKIP_NEXT=1; continue ;;
        -Xptxas=*|-Xcudafe=*|-Xnvlink=*|-Xarchive=*) continue ;;
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
    -x cuda \\${CUDA_DEVICE_FLAGS_STR} \\
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

if [[ ${TOOLKIT_ONLY} -eq 1 ]]; then
    exit 0
fi

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
    -DGGML_CUDA_FA=OFF \
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
