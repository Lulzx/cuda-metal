#!/bin/bash
# Verify that libcusolver.dylib and libcusparse.dylib symlinks exist
set -euo pipefail

LIB_DIR="$1"

for lib in libcusparse.dylib libcusolver.dylib libcublasLt.dylib; do
    target="$LIB_DIR/$lib"
    if [ ! -e "$target" ]; then
        echo "FAIL: $target does not exist"
        exit 1
    fi
    if [ ! -L "$target" ]; then
        echo "FAIL: $target is not a symlink"
        exit 1
    fi
    link_target=$(readlink "$target")
    if [ "$link_target" != "libcumetal.dylib" ]; then
        echo "FAIL: $target -> $link_target (expected libcumetal.dylib)"
        exit 1
    fi
done

echo "PASS: cusparse/cusolver/cublasLt library symlinks present"
