#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_CU="$4"
INCLUDE_DIR="$5"
OUTPUT_METALLIB="$6"

if [[ -z "${CUMETAL_CUDA_CLANG:-}" \
      && ! -x /opt/homebrew/opt/llvm/bin/clang++ \
      && ! -x /usr/local/opt/llvm/bin/clang++ ]]; then
  echo "SKIP: CUDA-capable Homebrew LLVM clang++ not available"
  exit 77
fi

if [[ -n "${CUMETAL_CUDA_CLANG:-}" ]]; then
  REAL_CUDA_CLANG="${CUMETAL_CUDA_CLANG}"
elif [[ -x /opt/homebrew/opt/llvm/bin/clang++ ]]; then
  REAL_CUDA_CLANG=/opt/homebrew/opt/llvm/bin/clang++
else
  REAL_CUDA_CLANG=/usr/local/opt/llvm/bin/clang++
fi
WRAPPER_DIR="$(mktemp -d)"
trap 'rm -rf "${WRAPPER_DIR}"' EXIT
cat >"${WRAPPER_DIR}/clang++" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
threshold=false
force_inline=false
previous=""
for argument in "$@"; do
  if [[ "${argument}" == "-fgpu-inline-threshold=1000000" ]]; then
    threshold=true
  fi
  if [[ "${previous}" == "-mllvm" && "${argument}" == "-inline-all-viable-calls" ]]; then
    force_inline=true
  fi
  previous="${argument}"
done
if [[ "${threshold}" != true || "${force_inline}" != true ]]; then
  echo "FAIL: CUDA frontend omitted forced viable-call inlining" >&2
  exit 64
fi
exec "${CUMETAL_FRONTEND_REAL_CLANG}" "$@"
EOF
chmod +x "${WRAPPER_DIR}/clang++"
export CUMETAL_FRONTEND_REAL_CLANG="${REAL_CUDA_CLANG}"
export CUMETAL_CUDA_CLANG="${WRAPPER_DIR}/clang++"

if "$CUMETALC" \
    --cuda-device \
    --cuda-arch invalid \
    --mode experimental \
    --input "$INPUT_CU" \
    --output "${OUTPUT_METALLIB}.invalid" \
    --entry cuda_device_probe \
    -I "$INCLUDE_DIR" \
    -D CUMETAL_FRONTEND_TEST_VALUE=2 \
    --overwrite >/dev/null 2>&1; then
  echo "FAIL: invalid CUDA architecture was accepted" >&2
  exit 1
fi

if "$CUMETALC" \
    --cuda-device \
    --cuda-inline-threshold invalid \
    --mode experimental \
    --input "$INPUT_CU" \
    --output "${OUTPUT_METALLIB}.invalid-threshold" \
    --entry cuda_device_probe \
    --overwrite >/dev/null 2>&1; then
  echo "FAIL: invalid CUDA inline threshold was accepted" >&2
  exit 1
fi

"$CUMETALC" \
  --cuda-device \
  --mode experimental \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --entry cuda_device_probe \
  --ptx-strict \
  --cuda-inline-threshold 1000000 \
  -I "$INCLUDE_DIR" \
  -D CUMETAL_FRONTEND_TEST_VALUE=2 \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUTPUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "cuda_device_probe"' >/dev/null

echo "PASS: cumetalc CUDA device frontend emit+validate completed"
