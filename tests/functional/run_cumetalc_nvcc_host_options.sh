#!/usr/bin/env bash
set -euo pipefail

compiler=${1:?compiler path required}
source_cu=${2:?source required}
work=${3:?work directory required}
clang=${CUMETAL_CUDA_CLANG:-${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}}
if [[ ! -x "$clang" ]]; then
    clang=$(command -v clang++ || true)
fi
if [[ -z "$clang" ]]; then
    echo "SKIP: no host Clang"
    exit 77
fi
mkdir -p "$work"
# A missing OpenMP runtime is a genuine prerequisite; a serial CUDA compile is not.
printf 'int main() { return 0; }\n' > "$work/openmp.cpp"
if ! "$clang" -fopenmp "$work/openmp.cpp" -o "$work/openmp" > "$work/prerequisite.log" 2>&1; then
    echo "SKIP: selected host Clang has no OpenMP runtime"
    cat "$work/prerequisite.log"
    exit 77
fi
export CUMETAL_CUDA_CLANG="$clang"
export OMP_DYNAMIC=FALSE

for form in separate equals long_separate long_equals; do
    case "$form" in
        separate) options=(-Xcompiler -fopenmp,-DHOST_OFFSET=17) ;;
        equals) options=(-Xcompiler=-fopenmp,-DHOST_OFFSET=17) ;;
        long_separate) options=(--compiler-options -fopenmp,-DHOST_OFFSET=17) ;;
        long_equals) options=(--compiler-options=-fopenmp,-DHOST_OFFSET=17) ;;
    esac
    for mode in object executable; do
        output="$work/$form-$mode"
        rm -f "$output" "$output.o"
        if [[ "$mode" == object ]]; then
            "$compiler" "$source_cu" "${options[@]}" -c -o "$output.o"
            "$compiler" "$output.o" "${options[@]}" -o "$output"
        else
            "$compiler" "$source_cu" "${options[@]}" -o "$output"
        fi
        "$output"
    done
done

for option in -Xcompiler --compiler-options -Xcompiler= --compiler-options=; do
    status=0
    "$compiler" "$source_cu" -o "$work/invalid" "$option" > "$work/invalid.log" 2>&1 || status=$?
    if [[ "$status" != 2 ]]; then
        cat "$work/invalid.log"
        echo "FAIL: $option must reject an absent host option with exit 2" >&2
        exit 1
    fi
done

# Exercise the second native link path too, with numerical GPU validation.
if command -v xcrun >/dev/null 2>&1 && xcrun metal --version >/dev/null 2>&1; then
    rm -f "$work/kernel"
    "$compiler" "$source_cu" -DWITH_KERNEL -Xcompiler=-fopenmp,-DHOST_OFFSET=17 -o "$work/kernel"
    CUMETAL_TRACE_GPU=1 "$work/kernel" > "$work/kernel.log" 2>&1
    cat "$work/kernel.log"
    grep -q 'device=apple_gpu' "$work/kernel.log"
    grep -Eq 'launch_success=true duration_ns=[1-9][0-9]*' "$work/kernel.log"
else
    echo "SKIP: kernel path needs offline Metal tools; host paths verified"
fi
