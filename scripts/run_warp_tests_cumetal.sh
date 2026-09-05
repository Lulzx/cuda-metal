#!/usr/bin/env bash
# run_warp_tests_cumetal.sh — run NVIDIA Warp's own test suite on CuMetal's
# device and print a tally.
#
# Usage:
#   bash scripts/run_warp_tests_cumetal.sh              # every test_*.py module
#   bash scripts/run_warp_tests_cumetal.sh test_vec ... # only these modules
#
# Environment overrides:
#   CUMETAL_WARP_DIR       Warp checkout (default: ../warp-cumetal)
#   CUMETAL_WARP_RESULTS   where per-module logs land (default: build/warp-tests)
#   CUMETAL_WARP_TIMEOUT   per-module wall clock cap in seconds (default: 300)
#
# ── why a module per process, and why a timeout ──────────────────────────────
# Warp's suite is nowhere near green on Metal yet, and the interesting failures
# are not assertion failures: a third of the modules take the interpreter down
# with them (SIGSEGV/SIGTRAP/SIGBUS), and at least one wedges the GPU into a
# command-buffer timeout. A single unittest process would stop at the first of
# those and report nothing about the rest, so each module gets its own process
# and its own deadline, and a module that dies is recorded as died.
#
# ── why cuda_0 only ──────────────────────────────────────────────────────────
# `-k cuda_0` selects the device-parameterised tests. The CPU half of the suite
# runs on Warp's bundled LLVM JIT with no CuMetal involvement at all, and it has
# its own macOS-arm64 crash (test_binary_ops_float16_cpu), which would otherwise
# be miscounted here as a CuMetal failure.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WARP_DIR="${CUMETAL_WARP_DIR:-${ROOT_DIR}/../warp-cumetal}"
RESULTS="${CUMETAL_WARP_RESULTS:-${ROOT_DIR}/build/warp-tests}"
TIMEOUT="${CUMETAL_WARP_TIMEOUT:-300}"

if [[ ! -f "${WARP_DIR}/warp/tests/__init__.py" ]]; then
    echo "ERROR: no Warp checkout at ${WARP_DIR}." >&2
    echo "       Run scripts/build_warp_cumetal.sh --build first." >&2
    exit 2
fi

PYTHON="${CUMETAL_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in python3 python; do
        command -v "${candidate}" >/dev/null 2>&1 && { PYTHON="${candidate}"; break; }
    done
fi
[[ -n "${PYTHON}" ]] || { echo "ERROR: no python interpreter on PATH." >&2; exit 2; }

modules=("$@")
if [[ ${#modules[@]} -eq 0 ]]; then
    for path in "${WARP_DIR}"/warp/tests/test_*.py; do
        modules+=("$(basename "${path}" .py)")
    done
fi

mkdir -p "${RESULTS}"
cd "${WARP_DIR}"

for module in "${modules[@]}"; do
    log="${RESULTS}/${module}.log"
    PYTHONPATH="${WARP_DIR}" "${PYTHON}" -m unittest -k cuda_0 \
        "warp.tests.${module}" >"${log}" 2>&1 &
    pid=$!
    ( sleep "${TIMEOUT}"; kill -9 "${pid}" 2>/dev/null ) &
    watchdog=$!
    wait "${pid}"
    echo "$?" >"${RESULTS}/${module}.exit"
    kill "${watchdog}" 2>/dev/null
    wait "${watchdog}" 2>/dev/null
done

# ── tally ────────────────────────────────────────────────────────────────────
# A module with no "Ran N tests" line never reached a unittest summary, which
# means it died rather than failed; those are counted separately because a
# crash hides however many tests it never got to.
"${PYTHON}" - "${RESULTS}" <<'PY'
import pathlib, re, sys

results = pathlib.Path(sys.argv[1])
run = fail = err = skip = 0
clean, died = [], []

for log in sorted(results.glob("*.log")):
    text = log.read_text(errors="replace")
    summary = re.search(r"^Ran (\d+) tests? in", text, re.M)
    if not summary:
        exit_file = log.with_suffix(".exit")
        code = exit_file.read_text().strip() if exit_file.exists() else "?"
        died.append(f"{log.stem}({code})")
        continue
    detail = re.search(r"^(?:OK|FAILED)(?: \((.*)\))?", text, re.M)
    detail = detail.group(1) if detail and detail.group(1) else ""
    counts = dict(re.findall(r"(failures|errors|skipped)=(\d+)", detail))
    n = int(summary.group(1))
    f, e, s = (int(counts.get(k, 0)) for k in ("failures", "errors", "skipped"))
    run, fail, err, skip = run + n, fail + f, err + e, skip + s
    if n and not f and not e:
        clean.append(log.stem)

total = len(clean) + len(died) + len(list(results.glob("*.log"))) - len(clean) - len(died)
print(f"modules: {len(list(results.glob('*.log')))} total, {len(died)} died before reporting")
print(f"tests:   {run} ran -> {run - fail - err - skip} pass, {fail} fail, {err} error, {skip} skip")
print(f"clean:   {', '.join(clean) if clean else '(none)'}")
if died:
    print(f"died:    {', '.join(died)}")
PY
