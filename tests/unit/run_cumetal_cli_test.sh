#!/usr/bin/env bash
set -euo pipefail

CUMETAL="${1:?usage: run_cumetal_cli_test.sh <cumetal> <runtime-directory>}"
RUNTIME_DIR="${2:?}"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

"$CUMETAL" version | grep -q '^cumetal '
"$CUMETAL" --help | grep -q 'cumetalc program.cu -o program'

# macOS strips DYLD_* when launching a platform-protected system binary. Build
# an ordinary user executable so the test observes the environment that a CUDA
# application receives.
cat > "$TMP_ROOT/printenv.c" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const char *value = getenv("DYLD_LIBRARY_PATH");
    if (value == NULL) return 1;
    puts(value);
    return 0;
}
EOF
/usr/bin/cc "$TMP_ROOT/printenv.c" -o "$TMP_ROOT/printenv"
DYLD_PATH="$("$CUMETAL" run "$TMP_ROOT/printenv")"
case "$DYLD_PATH" in
  "$RUNTIME_DIR"|"$RUNTIME_DIR":*) ;;
  *)
    echo "FAIL: cumetal run did not prepend the runtime directory: $DYLD_PATH" >&2
    exit 1
    ;;
esac

set +e
"$CUMETAL" run >/dev/null 2>&1
MISSING_STATUS=$?
"$CUMETAL" unknown >/dev/null 2>&1
UNKNOWN_STATUS=$?
"$CUMETAL" run /path/that/does/not/exist >/dev/null 2>&1
EXEC_STATUS=$?
set -e

if [[ "$MISSING_STATUS" -ne 2 ]]; then
  echo "FAIL: missing run target returned $MISSING_STATUS instead of 2" >&2
  exit 1
fi
if [[ "$UNKNOWN_STATUS" -ne 2 ]]; then
  echo "FAIL: unknown command returned $UNKNOWN_STATUS instead of 2" >&2
  exit 1
fi
if [[ "$EXEC_STATUS" -ne 127 ]]; then
  echo "FAIL: missing executable returned $EXEC_STATUS instead of 127" >&2
  exit 1
fi

echo "PASS: cumetal CLI reports its version and launches with the installed runtime"
