#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <cumetal-toolkit-dir> <work-dir>" >&2
    exit 2
fi

toolkit="$1"
work="$2"
nvcc="${toolkit}/bin/nvcc"

if [[ ! -x "${nvcc}" ]]; then
    echo "missing toolkit nvcc shim at ${nvcc}" >&2
    exit 1
fi

# The probes CMake performs first.
[[ "$("${nvcc}" --version)" == *"V12.0.140"* ]] || { echo "bad --version probe" >&2; exit 1; }
[[ "$("${nvcc}" -v __cmake_determine_cuda)" == *"TOP=${toolkit}"* ]] || {
    echo "bad -v TOP probe" >&2
    exit 1
}

rm -rf "${work}"
mkdir -p "${work}/src"
cat > "${work}/src/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.24)
project(cumetal_cmake_sample CUDA CXX)
add_executable(host_only host_only.cu)
add_executable(vadd main.cu)
EOF
cat > "${work}/src/host_only.cu" <<'EOF'
#include <cstdio>
int main() {
    std::printf("host-only ok\n");
    return 0;
}
EOF
cat > "${work}/src/main.cu" <<'EOF'
#include <cstdio>
__global__ void fill(int* x) { x[threadIdx.x] = (int)threadIdx.x; }
int main() {
    std::printf("vadd built\n");
    return 0;
}
EOF

cmake -S "${work}/src" -B "${work}/build" \
    -DCMAKE_CUDA_COMPILER="${nvcc}" > "${work}/configure.log" 2>&1 || {
    cat "${work}/configure.log" >&2
    echo "cmake configure failed" >&2
    exit 1
}
grep -q "The CUDA compiler identification is NVIDIA" "${work}/configure.log" || {
    cat "${work}/configure.log" >&2
    echo "CUDA compiler was not identified as NVIDIA" >&2
    exit 1
}

# The kernel-free target must compile, link and run everywhere: it drives
# the object and link modes of the nvcc-compatible driver.
cmake --build "${work}/build" --target host_only > "${work}/build-host.log" 2>&1 || {
    cat "${work}/build-host.log" >&2
    echo "host-only build failed" >&2
    exit 1
}
[[ "$("${work}/build/host_only")" == "host-only ok" ]] || {
    echo "host-only program produced wrong output" >&2
    exit 1
}

# Kernel targets need the offline Metal toolchain for metallib emission.
if command -v xcrun >/dev/null 2>&1 && xcrun metal --version >/dev/null 2>&1; then
    cmake --build "${work}/build" --target vadd > "${work}/build-vadd.log" 2>&1 || {
        cat "${work}/build-vadd.log" >&2
        echo "vadd build failed" >&2
        exit 1
    }
    [[ "$("${work}/build/vadd")" == "vadd built" ]] || {
        echo "vadd program produced wrong output" >&2
        exit 1
    }
else
    echo "SKIP: vadd build needs the Metal toolchain (configure + host-only verified)"
fi

echo "cmake toolkit drop-in ok"
