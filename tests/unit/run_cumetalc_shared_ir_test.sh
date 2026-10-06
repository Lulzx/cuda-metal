#!/usr/bin/env bash
set -euo pipefail

cumetalc=$1
ptx=$2
cu=$3
unsupported=$4
switch_source=$5
float_abs_source=$6
float_math_source=$7
byval_aggregate_memcpy_source=$8
float_atomic_add_source=$9
texture_vector_source=${10}
nvcc=${11}

workdir=$(mktemp -d "${TMPDIR:-/tmp}/cumetalc-shared-ir.XXXXXX")
trap 'rm -rf "$workdir"' EXIT

# The nvcc shim is generated into the build tree on demand, not by the build, so
# it is present only where something has already asked for it. This test passed
# in whichever tree happened to have one and failed everywhere else. Generate it
# here rather than depend on that.
if [[ ! -x "$nvcc" ]]; then
    toolkit_build_dir="${nvcc%/cumetal-cuda-toolkit/bin/nvcc}"
    root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    CUMETAL_BUILD_DIR="$toolkit_build_dir" \
        bash "$root_dir/scripts/build_llama_cpp_cumetal.sh" --toolkit-only >/dev/null
fi

"$cumetalc" "$ptx" --backend=cumetal-ir --emit=cumetal-ir \
    --overwrite -o "$workdir/vector.cmir"
grep -q 'kernel @vector_add' "$workdir/vector.cmir"
grep -q 'gpu.thread_id' "$workdir/vector.cmir"

"$cumetalc" "$ptx" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/vector.metal"
grep -q 'cumetal-provenance: generic_ptx_lowering' "$workdir/vector.metal"
grep -q 'cumetal-semantic-quality: exact' "$workdir/vector.metal"
grep -q 'kernel void vector_add' "$workdir/vector.metal"

"$cumetalc" "$cu" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/source.metal"
grep -q 'cumetal-provenance: generic_nvvm_lowering' "$workdir/source.metal"
grep -q 'kernel void vector_add' "$workdir/source.metal"

"$cumetalc" "$switch_source" --backend=cumetal-ir --emit=llvm \
    --overwrite -o "$workdir/switch.ll"
if grep -q ' switch ' "$workdir/switch.ll"; then
    echo "LLVM switch survived canonical CUDA normalization" >&2
    exit 1
fi
grep -q 'br i1' "$workdir/switch.ll"

"$cumetalc" "$float_abs_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/float_abs.metal"
grep -q 'kernel void cuda_float_abs' "$workdir/float_abs.metal"
test "$(grep -o 'fabs(' "$workdir/float_abs.metal" | wc -l | tr -d ' ')" -ge 2

"$cumetalc" "$byval_aggregate_memcpy_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/byval_aggregate_memcpy.metal"
grep -q 'kernel void byval_aggregate_memcpy' "$workdir/byval_aggregate_memcpy.metal"
grep -qE 'reinterpret_cast<device (cm_alias_)?uint\*>' "$workdir/byval_aggregate_memcpy.metal"

"$cumetalc" "$float_atomic_add_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/float_atomic_add.metal"
grep -q 'kernel void cuda_float_atomic_add' "$workdir/float_atomic_add.metal"
test "$(grep -o 'reinterpret_cast<device atomic_float\*>' "$workdir/float_atomic_add.metal" | wc -l | tr -d ' ')" -ge 2
if grep -qE 'cm_atomic_cas|atomic_compare_exchange' "$workdir/float_atomic_add.metal"; then
    echo "device float atomicAdd regressed to a CAS loop on the source-first path" >&2
    exit 1
fi

# Warp's texture.h instantiates tex1D/tex2D/tex3D for float2 and float4 with
# linear filtering; the software filter must accept vector texels.
"$cumetalc" "$texture_vector_source" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/texture_vector.metal"
grep -q 'kernel void cuda_texture_vector_fetch' "$workdir/texture_vector.metal"

# NVIDIA's C++ overlay selects binary32 for unsuffixed rsqrt/fma calls. A
# missing overload silently inserts f32->f64->f32 conversions and the double
# libdevice calls, which is especially expensive on Apple GPUs.
"$nvcc" -S --cuda-device-only -std=c++17 "$float_math_source" \
    -o "$workdir/float_math.ptx"
grep -q '__nv_rsqrtf' "$workdir/float_math.ptx"
grep -q '__nv_fmaf' "$workdir/float_math.ptx"
grep -q 'atom.global.add.f32' "$workdir/float_math.ptx"
if grep -q 'atom.*cas.b32' "$workdir/float_math.ptx"; then
    echo "float atomicAdd regressed to a CAS loop" >&2
    exit 1
fi
if grep -qE 'cvt\.f64\.f32|__nv_rsqrt([^fA-Za-z0-9_]|$)|__nv_fma([^fA-Za-z0-9_]|$)' \
    "$workdir/float_math.ptx"; then
    echo "FP32 CUDA math overloads promoted to FP64" >&2
    exit 1
fi

# Exercise the NVVM frontend too; the functional CUDA project tests PTX JIT.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
"$cumetalc" "$root_dir/tests/cuda_projects/kokkos_compat/kokkos_compat.cu" \
    --backend=cumetal-ir --emit=msl --overwrite -o "$workdir/kokkos_compat.metal"
grep -q 'cumetal-provenance: generic_nvvm_lowering' "$workdir/kokkos_compat.metal"
grep -q '= true;' "$workdir/kokkos_compat.metal"
grep -q '= false;' "$workdir/kokkos_compat.metal"

# Kokkos selects its CUDA implementation using these nvcc version macros.
cat > "$workdir/version.cu" <<'CU'
#if __CUDACC_VER_MAJOR__ != 12 || __CUDACC_VER_MINOR__ != 2 || __CUDACC_VER_BUILD__ != 140
#error inconsistent CUDA compiler version macros
#endif
__global__ void version_kernel(int* out) { *out = __CUDACC_VER_MAJOR__; }
CU
"$nvcc" -S --cuda-device-only "$workdir/version.cu" -o "$workdir/version.ptx"
grep -q 'version_kernel' "$workdir/version.ptx"
# CuPy's build_ext nvcc flags, verbatim apart from the paths.
"$nvcc" -c "$workdir/version.cu" -o "$workdir/version_cupy.o" \
    --generate-code=arch=compute_75,code=sm_75 \
    --generate-code=arch=compute_90,code=compute_90 \
    -Xfatbin=-compress-all -O2 '--compiler-options="-fPIC"' \
    --expt-relaxed-constexpr --std=c++17 -t2
test -s "$workdir/version_cupy.o"
# Kokkos forwards CMake's rpath as one comma-separated nvcc linker argument.
# Verify the executable links and carries the intended Mach-O load command.
cat > "$workdir/linker.cpp" <<'CPP'
int main() { return 0; }
CPP
"$nvcc" "$workdir/linker.cpp" -Xlinker "-rpath,$workdir" -Xlinker -rpath -Xlinker "$(dirname "$cumetalc")" -o "$workdir/linker"
"$workdir/linker"
otool -l "$workdir/linker" > "$workdir/load_commands.txt"
grep -F "path $workdir (offset" "$workdir/load_commands.txt"
"$nvcc" "$workdir/linker.cpp" "-Xlinker=-rpath,$workdir" "-Xlinker=-rpath,$(dirname "$cumetalc")" -o "$workdir/linker-equals"
"$workdir/linker-equals"


if "$cumetalc" "$unsupported" --backend=cumetal-ir --emit=msl \
    --overwrite -o "$workdir/unsupported.metal" \
    >"$workdir/unsupported.stdout" 2>"$workdir/unsupported.stderr"; then
    echo "unsupported PTX unexpectedly compiled" >&2
    exit 1
fi
grep -q 'unsupported opcode' "$workdir/unsupported.stderr"
test ! -e "$workdir/unsupported.metal"
