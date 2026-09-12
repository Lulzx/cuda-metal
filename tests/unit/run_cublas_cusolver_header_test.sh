#!/usr/bin/env bash
# cublas_v2.h and cusolverDn.h share cublasFillMode_t / cublasSideMode_t. They
# used to each define the enums, so a TU including both failed to compile.
# Both include orders must work, and the canonical enum keeps
# CUBLAS_FILL_MODE_FULL for dense-solver callers.
set -euo pipefail

CXX_COMPILER="$1"
INCLUDE_DIR="$2"
WORK_DIR="$3"

if [[ ! -x "$CXX_COMPILER" ]]; then
  echo "SKIP: C++ compiler not executable at $CXX_COMPILER"
  exit 77
fi

if [[ ! -d "$INCLUDE_DIR" ]]; then
  echo "FAIL: include directory missing"
  exit 1
fi

mkdir -p "$WORK_DIR"

cat >"$WORK_DIR/header_order_blas_first.cpp" <<'EOF'
#include <cublas_v2.h>
#include <cusolverDn.h>
static_assert(CUBLAS_FILL_MODE_LOWER == 0, "LOWER");
static_assert(CUBLAS_FILL_MODE_UPPER == 1, "UPPER");
static_assert(CUBLAS_FILL_MODE_FULL == 2, "FULL missing");
static_assert(CUBLAS_SIDE_LEFT == 0 && CUBLAS_SIDE_RIGHT == 1, "side modes");
int main() { return 0; }
EOF

cat >"$WORK_DIR/header_order_solver_first.cpp" <<'EOF'
#include <cusolverDn.h>
#include <cublas_v2.h>
static_assert(CUBLAS_FILL_MODE_LOWER == 0, "LOWER");
static_assert(CUBLAS_FILL_MODE_UPPER == 1, "UPPER");
static_assert(CUBLAS_FILL_MODE_FULL == 2, "FULL missing");
static_assert(CUBLAS_SIDE_LEFT == 0 && CUBLAS_SIDE_RIGHT == 1, "side modes");
int main() { return 0; }
EOF

for src in header_order_blas_first.cpp header_order_solver_first.cpp; do
  "$CXX_COMPILER" \
    -std=c++17 \
    -I"$INCLUDE_DIR" \
    -fsyntax-only \
    "$WORK_DIR/$src"
done

echo "PASS: cublas_v2.h and cusolverDn.h co-include in both orders"
