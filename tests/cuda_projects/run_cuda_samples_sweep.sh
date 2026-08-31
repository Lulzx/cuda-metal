#!/usr/bin/env bash
# Build and run upstream NVIDIA cuda-samples sources, unmodified, against libcumetal
# and compare each one's outcome to tests/cuda_projects/cuda_samples_sweep_manifest.txt.
#
# This exists because CuMetal's own test suite cannot see a whole class of defect.
# The samples reach for CUDA surface that in-repo tests never touch -- most memorably
# the canonical include-guard macros (__DRIVER_TYPES_H__ and friends) that
# Common/helper_cuda.h feature-detects on, whose absence made 82 of 88 samples fail
# to compile while every CuMetal test stayed green.
#
# Usage: run_cuda_samples_sweep.sh <cumetal-root> <ctest-binary-dir> [manifest]
set -uo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
MANIFEST="${3:-${ROOT_DIR}/tests/cuda_projects/cuda_samples_sweep_manifest.txt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${SCRIPT_DIR}/_common.sh"
# shellcheck source=tests/cuda_projects/sweep_status.sh
source "${SCRIPT_DIR}/sweep_status.sh"

CUDA_SAMPLES_DIR="${CUMETAL_CUDA_SAMPLES_DIR:-${ROOT_DIR}/../cuda-samples}"

# Upstream moved the C++ samples from Samples/ to cpp/ . Support both so the sweep
# runs against whichever checkout is on the machine.
CATEGORY_ROOT=""
for candidate in "${CUDA_SAMPLES_DIR}/cpp" "${CUDA_SAMPLES_DIR}/Samples"; do
    if [[ -d "${candidate}/0_Introduction/vectorAdd" ]]; then
        CATEGORY_ROOT="${candidate}"
        break
    fi
done
if [[ -z "${CATEGORY_ROOT}" ]]; then
    echo "SKIP: no cuda-samples checkout at ${CUDA_SAMPLES_DIR}"
    echo "      Clone https://github.com/NVIDIA/cuda-samples or set CUMETAL_CUDA_SAMPLES_DIR."
    exit 77
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "FAIL: manifest not found at ${MANIFEST}"
    exit 1
fi
if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

COMMON_DIR="${CUDA_SAMPLES_DIR}/Common"
OUT_ROOT="${BUILD_DIR}/cuda_samples_sweep"
mkdir -p "${OUT_ROOT}"

source "${ROOT_DIR}/scripts/cumetal_cuda_flags.sh"
cumetal_cuda_device_flags
export PATH="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cuda_toolchain:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}"
CUMETAL_LIB_DIR="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"

# macOS ships no coreutils `timeout`; a sample that hangs must not hang the suite.
run_with_timeout() {
    local secs="$1"; shift
    "$@" & local pid=$!
    (
        local waited=0
        while (( waited < secs * 10 )); do
            kill -0 "$pid" 2>/dev/null || exit 0
            /bin/sleep 0.1
            waited=$((waited + 1))
        done
        kill -9 "$pid" 2>/dev/null
    ) & local watcher=$!
    local rc=0
    wait "$pid" || rc=$?
    kill -9 "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    (( rc == 137 )) && rc=124
    return $rc
}

# Classify one sample. Echoes exactly one of the manifest status words.
classify_sample() {
    local rel="$1"
    local name src_dir out_dir log
    name="$(basename "${rel}")"
    src_dir="${CATEGORY_ROOT}/${rel}"
    out_dir="${OUT_ROOT}/${name}"
    log="${out_dir}/build-and-run.log"

    if [[ ! -d "${src_dir}" ]]; then
        echo "absent"
        return
    fi
    mkdir -p "${out_dir}"
    : > "${log}"

    local objs=() sources=() f base obj
    shopt -s nullglob
    sources=( "${src_dir}"/*.cu "${src_dir}"/*.cpp )
    shopt -u nullglob
    if (( ${#sources[@]} == 0 )); then
        echo "absent"
        return
    fi

    # First use the ordinary source-first compilation path. If Clang diagnoses
    # a device-side kernel launch, retry the complete translation unit set with
    # relocatable device code and the legacy offload driver (the new driver
    # emits an invalid Mach-O section for RDC on macOS). This is capability-
    # driven rather than sample-name-driven, so any CUDA source using dynamic
    # parallelism follows the same path.
    local compile_mode compile_ok=0
    local llvm_objcopy="${CUMETAL_LLVM_OBJCOPY:-/opt/homebrew/opt/llvm/bin/llvm-objcopy}"
    for compile_mode in ordinary rdc; do
        objs=()
        compile_ok=1
        for f in "${sources[@]}"; do
            base="$(basename "${f}")"
            obj="${out_dir}/${base}.o"
            rm -f "${obj}"
            local lang=()
            if [[ "${base}" == *.cu ]]; then
                lang=( -x cuda "${CUMETAL_CUDA_DEVICE_FLAGS[@]}" -nocudainc -nocudalib
                       -D__CUDACC__=1 -D__NVCC__=1
                       -fgpu-inline-threshold=1000000
                       -mllvm -inline-all-viable-calls )
                if [[ "${compile_mode}" == rdc ]]; then
                    lang+=( --no-offload-new-driver -fgpu-rdc )
                fi
            fi
            # -Wno-everything: upstream sample warnings are not this test's business.
            # Compile failures still surface through the exit status.
            if ! run_with_timeout 300 "${CLANG_BIN}" "${lang[@]}" \
                    -std=c++17 -O2 -DNDEBUG -Wno-everything -Wno-pass-failed \
                    -I"${ROOT_DIR}/runtime/api" -I"${COMMON_DIR}" -I"${src_dir}" \
                    -include cuda_runtime.h -include new \
                    -c "${f}" -o "${obj}" >>"${log}" 2>&1; then
                compile_ok=0
                break
            fi
            if [[ "${compile_mode}" == rdc && "${base}" == *.cu ]]; then
                if [[ ! -x "${llvm_objcopy}" ]]; then
                    echo "missing llvm-objcopy for CUDA RDC linked-binary registration" >>"${log}"
                    compile_ok=0
                    break
                fi
                local linked_symbol
                linked_symbol="$(nm -u "${obj}" 2>/dev/null | sed -n \
                    's/^[[:space:]]*\(___cudaRegisterLinkedBinary__nv_[[:alnum:]_]*\)$/\1/p' | head -1)"
                if [[ -n "${linked_symbol}" ]]; then
                    "${llvm_objcopy}" \
                        --redefine-sym "${linked_symbol}=___cudaRegisterLinkedBinary" \
                        "${obj}" >>"${log}" 2>&1 || {
                            compile_ok=0
                            break
                        }
                fi
                # Clang 23 may give every RDC translation unit the same defined
                # fatbin wrapper symbol. The wrappers are TU-local registration
                # objects, so make each definition unique before the host link.
                local fatbin_wrapper unique_wrapper
                fatbin_wrapper="$(nm -g "${obj}" 2>/dev/null | sed -n \
                    's/^[[:xdigit:]]*[[:space:]][[:alpha:]][[:space:]]\(___fatbinwrap__nv_[[:alnum:]_]*\)$/\1/p' | head -1)"
                if [[ -n "${fatbin_wrapper}" ]]; then
                    unique_wrapper="${fatbin_wrapper}_$(printf '%s' "${base}" | tr -c '[:alnum:]_' '_')"
                    "${llvm_objcopy}" \
                        --redefine-sym "${fatbin_wrapper}=${unique_wrapper}" \
                        "${obj}" >>"${log}" 2>&1 || {
                            compile_ok=0
                            break
                        }
                fi
            fi
            objs+=( "${obj}" )
        done
        (( compile_ok == 1 )) && break
        echo "retrying all sources with CUDA relocatable device code" >>"${log}"
    done
    if (( compile_ok == 0 )); then
        echo "compile-fail"
        return
    fi

    local link_extra=()
    if [[ "${rel}" == "0_Introduction/simpleTexture3D" ]]; then
        link_extra=( -framework OpenGL -framework GLUT )
    fi
    if ! xcrun clang++ "${objs[@]}" "${link_extra[@]}" \
            -L"${CUMETAL_LIB_DIR}" -lcumetal -Wl,-rpath,"${CUMETAL_LIB_DIR}" \
            -o "${out_dir}/${name}" >>"${log}" 2>&1; then
        echo "link-fail"
        return
    fi

    # Several samples read data files relative to their source directory.
    local run_output status=0
    local run_args=()
    if [[ "${rel}" == "0_Introduction/simpleTexture3D" ]]; then
        run_args=( --file data/ref_texture3D.bin )
    elif [[ "${rel}" == "6_Performance/UnifiedMemoryPerf" ]]; then
        # The upstream default repeats every 64 MiB allocation/transfer mode
        # twenty times. One iteration still runs and numerically verifies the
        # complete mode matrix while keeping a conformance sweep below the
        # Metal GPU watchdog; this gate measures correctness, not throughput.
        run_args=( --kernel-iterations=1 )
    fi
    run_output="$(cd "${src_dir}" && run_with_timeout 120 "${out_dir}/${name}" "${run_args[@]}" 2>&1)" || status=$?
    printf '%s\n' "${run_output}" >>"${log}"

    if (( status == 124 )); then
        echo "timeout"
        return
    fi
    if grep -q "registered kernel missing metallib" <<<"${run_output}"; then
        echo "no-lowering"
        return
    fi
    # simpleCudaGraphs has no built-in numerical assertion: it exits zero after
    # printing each reduction. All twelve launches (manual graph, clone, captured
    # graph, clone; three iterations each) consume identical input and must agree.
    # Treating process exit alone as ground truth previously green-washed values
    # ranging from zero to several times the expected sum.
    if [[ "${rel}" == "3_CUDA_Features/simpleCudaGraphs" ]]; then
        local graph_values
        graph_values="$(sed -n 's/.*final reduced sum = \([-+0-9.eE]*\).*/\1/p' <<<"${run_output}")"
        if ! awk '
            NR == 1 { min = max = $1 + 0.0 }
            { value = $1 + 0.0; if (value < min) min = value; if (value > max) max = value }
            END {
                scale = (max < 0 ? -max : max); if (scale < 1.0) scale = 1.0;
                exit !(NR == 12 && max > 0.0 && (max - min) <= 1.0e-5 * scale)
            }
        ' <<<"${graph_values}"; then
            echo "run-fail"
            return
        fi
    fi
    # This sample's only numerical oracle is device printf. It otherwise exits
    # zero even when a cached kernel is launched without its hidden printf ABI,
    # which previously let a no-op run look like conformance.
    if [[ "${rel}" == "0_Introduction/simpleCooperativeGroups" ]]; then
        local whole_group_lines tiled_group_lines
        whole_group_lines="$(grep -c 'threadBlockGroup is 2016 (expected 2016)' <<<"${run_output}")"
        tiled_group_lines="$(grep -c 'tiledPartition16 group is 120 (expected 120)' <<<"${run_output}")"
        if [[ "${whole_group_lines}" != "1" || "${tiled_group_lines}" != "4" ]]; then
            echo "run-fail"
            return
        fi
    fi
    # Upstream exits from this sample using only the device-reduced residual.
    # A broken warp-tile reduction once sampled just the first warp, reported a
    # tiny residual, and exited success while the sample's independent host
    # equation check printed Error amount = 1.0. Require that host oracle too.
    if [[ "${rel}" == "4_CUDA_Libraries/conjugateGradientMultiBlockCG" ]]; then
        local equation_error
        equation_error="$(sed -n \
            's/.*Test Summary:  Error amount = \([-+0-9.eE]*\).*/\1/p' \
            <<<"${run_output}" | tail -1)"
        if ! awk 'BEGIN { value = ARGV[1] + 0.0; exit !(ARGV[1] != "" && value >= 0.0 && value <= 1.0e-4) }' \
            "${equation_error}"; then
            echo "run-fail"
            return
        fi
    fi
    # EXIT_WAIVED. The sample itself decided a capability it needs is absent and
    # declined to run -- the intended outcome when CuMetal reports it unsupported.
    if (( status == 2 )); then
        echo "waive"
        return
    fi
    if (( status != 0 )); then
        echo "run-fail"
        return
    fi
    if grep -Eq 'Test failed|Result = FAIL|FAILED|FAILURE|ERROR!' <<<"${run_output}"; then
        echo "run-fail"
        return
    fi
    echo "pass"
}

declare -a REGRESSIONS=() IMPROVEMENTS=() EVIDENCE_UPDATES=()
declare -i total=0 present=0 matched=0 gated_total=0 gated_present=0

echo "cuda-samples sweep against ${CATEGORY_ROOT}"
echo

while read -r expected rel; do
    [[ -z "${expected}" || "${expected}" == \#* ]] && continue
    total+=1
    [[ "${expected}" == "pass" || "${expected}" == "waive" ]] && gated_total+=1
    actual="$(classify_sample "${rel}")"

    if [[ "${actual}" == "absent" ]]; then
        printf '  %-12s %-58s (not in this checkout)\n' "SKIP" "${rel}"
        continue
    fi
    present+=1
    [[ "${expected}" == "pass" || "${expected}" == "waive" ]] && gated_present+=1
    if [[ "${actual}" == "${expected}" ]]; then
        matched+=1
        printf '  %-12s %-58s\n' "${actual}" "${rel}"
        continue
    fi

    printf '  %-12s %-58s (expected %s)\n' "${actual}" "${rel}" "${expected}"
    case "$(cumetal_sweep_transition "${expected}" "${actual}")" in
        regression)
            REGRESSIONS+=( "${rel}: expected ${expected}, got ${actual}" )
            ;;
        improvement)
            IMPROVEMENTS+=( "${rel}: ${expected} -> ${actual}" )
            ;;
        evidence-update)
            EVIDENCE_UPDATES+=( "${rel}: ${expected} -> ${actual}" )
            ;;
    esac
done < "${MANIFEST}"

echo
echo "${present}/${total} manifest samples present in this checkout; ${matched} matched."

# A partial checkout is not a pass. Without this, a sparse clone that contains none
# of the samples sails through every check below and reports success -- the sweep
# would certify a baseline it never ran. Require most of the gated samples to be
# here, or say plainly that the sweep could not run.
if (( gated_total > 0 && gated_present * 5 < gated_total * 4 )); then
    echo
    echo "SKIP: only ${gated_present}/${gated_total} gated samples are in this checkout."
    echo "      ${CUDA_SAMPLES_DIR} looks partial. Clone the full"
    echo "      https://github.com/NVIDIA/cuda-samples or set CUMETAL_CUDA_SAMPLES_DIR."
    exit 77
fi

if (( ${#REGRESSIONS[@]} > 0 )); then
    echo
    echo "FAIL: ${#REGRESSIONS[@]} sample(s) regressed:"
    printf '  - %s\n' "${REGRESSIONS[@]}"
    echo "Logs: ${OUT_ROOT}/<sample>/build-and-run.log"
    exit 1
fi

if (( ${#EVIDENCE_UPDATES[@]} > 0 )); then
    echo
    echo "FAIL: ${#EVIDENCE_UPDATES[@]} previously unrun sample(s) now have runtime evidence:"
    printf '  - %s\n' "${EVIDENCE_UPDATES[@]}"
    echo "Update ${MANIFEST} and docs/known-gaps.md with the observed classifications."
    exit 1
fi

if (( ${#IMPROVEMENTS[@]} > 0 )); then
    echo
    echo "FAIL: ${#IMPROVEMENTS[@]} sample(s) now succeed but the manifest still lists them"
    echo "as unsupported. This is good news -- record it so the unsupported set stays honest:"
    printf '  - %s\n' "${IMPROVEMENTS[@]}"
    echo "Update ${MANIFEST} and docs/known-gaps.md."
    exit 1
fi

echo "PASS: cuda-samples sweep matches the recorded baseline."
