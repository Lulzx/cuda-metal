#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <source-root> <cumetalc> <manifest>" >&2
    exit 2
fi

source_root="$1"
cumetalc="$2"
manifest="$3"

if [[ ! -x "${cumetalc}" ]]; then
    echo "SKIP: cumetalc is not built: ${cumetalc}" >&2
    exit 77
fi
if ! xcrun -f metal >/dev/null 2>&1 || ! xcrun -f metallib >/dev/null 2>&1; then
    echo "SKIP: Apple metal/metallib tools are unavailable" >&2
    exit 77
fi

matrix_tmp="$(mktemp -d "${TMPDIR:-/tmp}/cumetal-backend-matrix.XXXXXX")"
trap 'rm -rf "${matrix_tmp}"' EXIT HUP INT TERM

find "${source_root}/samples" "${source_root}/tests/cuda_projects" \
    -type f -name '*.cu' -print \
    | sed "s#^${source_root}/##" \
    | sort >"${matrix_tmp}/actual-corpus.txt"
awk 'NF >= 5 && $1 !~ /^#/ { print $1 }' "${manifest}" \
    | sort >"${matrix_tmp}/manifest-corpus.txt"
if ! cmp -s "${matrix_tmp}/actual-corpus.txt" "${matrix_tmp}/manifest-corpus.txt"; then
    echo "FAIL: backend matrix manifest does not exactly cover samples/ and tests/cuda_projects/" >&2
    diff -u "${matrix_tmp}/manifest-corpus.txt" "${matrix_tmp}/actual-corpus.txt" >&2 || true
    exit 1
fi

direct_legacy_pass=0
direct_ir_pass=0
ptx_legacy_pass=0
ptx_ir_pass=0
total=0
mismatches=0

run_cell() {
    local source_path="$1"
    local frontend="$2"
    local backend="$3"
    local expected="$4"
    local cell="$5"
    local safe_name output log observed diagnostic
    local -a args

    safe_name="$(printf '%s' "${source_path}" | tr '/.' '__')"
    output="${matrix_tmp}/${safe_name}-${cell}.metallib"
    log="${matrix_tmp}/${safe_name}-${cell}.log"
    args=("${source_root}/${source_path}")
    if [[ "${frontend}" == "ptx" ]]; then
        args+=(--cuda-device)
    fi
    args+=(--backend="${backend}" --emit=metallib --no-link --overwrite
           --fp64=fast48 -o "${output}")

    if "${cumetalc}" "${args[@]}" >"${log}" 2>&1; then
        observed=pass
        diagnostic=-
    else
        observed=fail
        diagnostic="$(grep -E 'cumetalc failed:| error:' "${log}" | tail -1 || true)"
        [[ -n "${diagnostic}" ]] || diagnostic="command failed"
        diagnostic="$(printf '%s' "${diagnostic}" | tr '\t\r\n' '   ')"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${source_path}" "${frontend}" "${backend}" "${expected}" "${observed}" "${diagnostic}"
    if [[ "${observed}" != "${expected}" ]]; then
        echo "FAIL: ${source_path} ${cell}: expected ${expected}, observed ${observed}" >&2
        sed -n '1,160p' "${log}" >&2
        mismatches=$((mismatches + 1))
    fi

    if [[ "${observed}" == pass ]]; then
        case "${cell}" in
            direct-legacy) direct_legacy_pass=$((direct_legacy_pass + 1)) ;;
            direct-cumetal-ir) direct_ir_pass=$((direct_ir_pass + 1)) ;;
            ptx-legacy) ptx_legacy_pass=$((ptx_legacy_pass + 1)) ;;
            ptx-cumetal-ir) ptx_ir_pass=$((ptx_ir_pass + 1)) ;;
        esac
    fi
}

printf 'source\tfrontend\tbackend\texpected\tobserved\tdiagnostic\n'
while read -r source_path direct_legacy direct_ir ptx_legacy ptx_ir extra; do
    [[ -z "${source_path}" || "${source_path}" == \#* ]] && continue
    if [[ -n "${extra:-}" ]]; then
        echo "FAIL: malformed backend matrix row for ${source_path}" >&2
        exit 1
    fi
    for expected in "${direct_legacy}" "${direct_ir}" "${ptx_legacy}" "${ptx_ir}"; do
        if [[ "${expected}" != pass && "${expected}" != fail ]]; then
            echo "FAIL: invalid expected status '${expected}' for ${source_path}" >&2
            exit 1
        fi
    done

    total=$((total + 1))
    run_cell "${source_path}" direct legacy "${direct_legacy}" direct-legacy
    run_cell "${source_path}" direct cumetal-ir "${direct_ir}" direct-cumetal-ir
    run_cell "${source_path}" ptx legacy "${ptx_legacy}" ptx-legacy
    run_cell "${source_path}" ptx cumetal-ir "${ptx_ir}" ptx-cumetal-ir
done <"${manifest}"

echo "backend_matrix_total=${total} direct_legacy=${direct_legacy_pass}/${total} direct_cumetal_ir=${direct_ir_pass}/${total} ptx_legacy=${ptx_legacy_pass}/${total} ptx_cumetal_ir=${ptx_ir_pass}/${total}"
if [[ ${mismatches} -ne 0 ]]; then
    echo "FAIL: ${mismatches} backend matrix classification(s) changed" >&2
    exit 1
fi
echo "PASS: compiler backend production-metallib matrix matches the reviewed manifest"
