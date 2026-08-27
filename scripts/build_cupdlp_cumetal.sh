#!/usr/bin/env bash
# build_cupdlp_cumetal.sh — Build cuPDLP-C (the LP solver HiGHS vendors as its
# GPU PDLP path) against CuMetal, and optionally compare it to a CPU build.
#
# Usage:
#   bash scripts/build_cupdlp_cumetal.sh [--compare]
#
# Environment overrides:
#   CUMETAL_CUPDLP_DIR   checkout path (default: $CLAUDE_JOB_DIR/tmp or /tmp)
#   CUMETAL_CUPDLP_REPO  git remote (default: https://github.com/COPT-Public/cuPDLP-C)
#   HIGHS_HOME           HiGHS install prefix (default: brew --prefix highs)
#   CUMETAL_JOBS         build parallelism (default: 2 -- higher can wedge a laptop
#                        when a test suite is running at the same time)
#
# cuPDLP-C needs three build patches that have nothing to do with Metal; they
# are applied here rather than upstreamed because they are all consequences of
# building an old standalone cuPDLP-C against a current HiGHS:
#
#   1. HiGHS >= 1.7 vendors its own cuPDLP-C copy and exports the same C symbols
#      (Init_Scaling, LP_SolvePDHG, cupdlp_*). mps_highs.c is a plain object in
#      the executable, so those calls bind to whichever dylib comes first and
#      binding them to HiGHS's copy segfaults immediately. libcupdlp must be
#      named by path so it stays ahead of libhighs.
#   2. HiGHS 1.15's HConst.h defines `enum ConstraintType { EQ, LEQ, GEQ, BOUND }`
#      at namespace scope, colliding with wrapper_highs.h's identical names.
#   3. cupdlp/CMakeLists.txt hardcodes /usr/local/cuda/include, so the CUDA
#      headers have to be put on the C/C++ flags explicitly.
#
# Patch 4 is a different kind of thing and is kept separate on purpose: it is an
# upstream correctness bug rather than a build-compatibility shim. It is not
# Metal-related either. See the comment on it below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOBS="${CUMETAL_JOBS:-2}"
COMPARE=0
[[ "${1:-}" == "--compare" ]] && COMPARE=1

DEFAULT_PARENT="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "${DEFAULT_PARENT}" 2>/dev/null || DEFAULT_PARENT=/tmp
CUPDLP_DIR="${CUMETAL_CUPDLP_DIR:-${DEFAULT_PARENT}/cuPDLP-C}"
CUPDLP_REPO="${CUMETAL_CUPDLP_REPO:-https://github.com/COPT-Public/cuPDLP-C}"

if [[ -z "${HIGHS_HOME:-}" ]]; then
    HIGHS_HOME="$(brew --prefix highs 2>/dev/null || true)"
fi
if [[ ! -d "${HIGHS_HOME}/include/highs" ]]; then
    echo "ERROR: HiGHS not found. Install with: brew install highs" >&2
    exit 2
fi
export HIGHS_HOME

FAKE_CUDA="${ROOT_DIR}/build/cumetal-cuda-toolkit"
if [[ ! -x "${FAKE_CUDA}/bin/nvcc" ]]; then
    echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing." >&2
    echo "       Run scripts/build_llama_cpp_cumetal.sh once to generate the toolkit." >&2
    exit 2
fi
# cuPDLP-C links cusparse, which the llama.cpp toolkit generator does not symlink.
for libdir in lib lib64; do
    for lib in cusparse cusolver cublasLt cumetal; do
        src="${ROOT_DIR}/build/lib${lib}.dylib"
        [[ -f "${src}" ]] && ln -sf "${src}" "${FAKE_CUDA}/${libdir}/lib${lib}.dylib"
    done
done
export CUDA_HOME="${FAKE_CUDA}"
export PATH="${FAKE_CUDA}/bin:${PATH}"
export DYLD_LIBRARY_PATH="${ROOT_DIR}/build:${DYLD_LIBRARY_PATH:-}"

if [[ ! -d "${CUPDLP_DIR}" ]]; then
    echo "Cloning cuPDLP-C -> ${CUPDLP_DIR} ..."
    git clone --depth 1 "${CUPDLP_REPO}" "${CUPDLP_DIR}"
fi
cd "${CUPDLP_DIR}"

# ── patch 1+2: modern-HiGHS compatibility ────────────────────────────────────
if ! grep -q CupdlpConstraintType interface/wrapper_highs.h 2>/dev/null; then
    echo "Patching cuPDLP-C for HiGHS >= 1.7 ..."
    /usr/bin/sed -i '' 's/} ConstraintType;/} CupdlpConstraintType;/' interface/wrapper_highs.h
    python3 - <<'PY'
import re
pat = re.compile(r'\b(EQ|LEQ|GEQ|BOUND)\b')
for f in ('interface/wrapper_highs.h', 'interface/wrapper_highs.cpp'):
    s = open(f).read()
    open(f, 'w').write(pat.sub(lambda m: 'CUPDLP_' + m.group(1), s))

p = 'interface/CMakeLists.txt'
s = open(p).read()
s = s.replace("""target_link_libraries(
        wrapper_lp PUBLIC cupdlp
)""", """target_link_libraries(
        wrapper_lp PUBLIC cupdlp ${CUDA_LIBRARY}
)""")
s = s.replace("""target_link_libraries(
        wrapper_highs PUBLIC cupdlp
        ${HiGHS_LIBRARY}
)""", """target_link_libraries(
        wrapper_highs PUBLIC cupdlp
        ${HiGHS_LIBRARY}
        ${CUDA_LIBRARY}
)""")
s = s.replace("""target_link_libraries(
        plc PUBLIC
        wrapper_highs
        ${HiGHS_LIBRARY}
)""", """# libcupdlp by path, so it stays ahead of libhighs's vendored copy of the
# same C symbols. A bare `cupdlp` target gets sorted after wrapper_highs.
target_link_libraries(
        plc PUBLIC
        ${CMAKE_BINARY_DIR}/lib/libcupdlp.dylib
        wrapper_highs
        ${HiGHS_LIBRARY}
)
add_dependencies(plc cupdlp)""")
open(p, 'w').write(s)
PY
fi

# ── patch 4: upstream out-of-bounds read in the power method ─────────────
# PDHG_Power_Method takes the squared norm of `ax` over nCols elements, but
# vec_Alloc sized `ax` to nRows (cupdlp_utils.c). Those agree only on a square
# LP, so on anything else it reads the wrong length, and which way it goes
# depends on the shape:
#
#   datt256   11,078 x 262,144   reads ~251k doubles past the end of ax
#   ex10      69,609 x 17,680    stays in bounds, norms a quarter of the vector
#
# The first segfaults under CuMetal, where every cudaMalloc is its own MTLBuffer
# and the page after one is usually unmapped; it cost two of five runs on
# datt256. The second is quieter: nothing crashes and the printed power-method
# residual is simply wrong (10637.216 against the correct 35847.690 at
# iteration 0).
#
# The step size the power method returns is not affected either way -- that one
# norms `aty`, which really is nCols long -- so this changes crash rate and a
# diagnostic, not the solve. It reproduces without CuMetal: the CPU build takes
# the same line, and only CUDA's slab allocator hides the read.
if ! grep -q "vec_Alloc sized to nRows" cupdlp/cupdlp_step.c 2>/dev/null; then
    echo "Patching cuPDLP-C power-method norm length ..."
    python3 - <<'PY'
p = 'cupdlp/cupdlp_step.c'
s = open(p).read()
old = "    cupdlp_twoNormSquared(work, lp->nCols, ax->data, &res);"
new = ("    // ax is vec_Alloc sized to nRows, not nCols; upstream reads nCols.\n"
       "    cupdlp_twoNormSquared(work, lp->nRows, ax->data, &res);")
assert s.count(old) == 1, "power-method norm line not found; upstream may have fixed it"
open(p, 'w').write(s.replace(old, new))
PY
fi

# ── configure + build ────────────────────────────────────────────────────────
echo "Building GPU (CuMetal) build ..."
cmake -S . -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="-I${CUDA_HOME}/include" \
      -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" >/dev/null
cmake --build build -j"${JOBS}" 2>&1 | grep -E "error|Error" && exit 1
echo "  -> ${CUPDLP_DIR}/build/bin/plc"

if [[ ${COMPARE} -eq 1 ]]; then
    echo "Building CPU reference build ..."
    cmake -S . -B build-cpu -DBUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build-cpu -j"${JOBS}" 2>&1 | grep -E "error|Error" && exit 1
    echo "  -> ${CUPDLP_DIR}/build-cpu/bin/plc"
fi

echo
echo "Run:  DYLD_LIBRARY_PATH=${ROOT_DIR}/build ${CUPDLP_DIR}/build/bin/plc -fname ${CUPDLP_DIR}/example/afiro.mps"
