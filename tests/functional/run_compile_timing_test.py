#!/usr/bin/env python3
"""Per-phase compile timing and the MSL size limit (#220), through cumetalc
(ahead of time) and the runtime (Metal source compiled in-process).

usage: run_compile_timing_test.py <cumetalc> <libcumetal.dylib>"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CUDA = r"""
#include <cstdio>
__global__ void timing_probe_kernel(float* x, float s) { x[threadIdx.x] *= s; }
int main() {
#if !defined(__CUMETAL__) || __CUMETAL_VERSION__ < 800
#error "cumetalc must define __CUMETAL__ and __CUMETAL_VERSION__"
#endif
    float* d = nullptr;
    if (cudaMalloc(&d, 32 * sizeof(float)) != cudaSuccess) return 1;
    timing_probe_kernel<<<1, 32>>>(d, 2.0f);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    std::printf("CUMETAL_VERSION %d\n", __CUMETAL_VERSION__);
    return 0;
}
"""

MSL = """#include <metal_stdlib>
using namespace metal;
kernel void jit_timing_probe(device uint* output [[buffer(0)]],
                             uint lane [[thread_position_in_grid]]) {
    output[lane] = lane * 3u + 7u;
}
"""


def base_env():
    return {k: v for k, v in os.environ.items()
            if k not in ("CUMETAL_COMPILE_TIMING", "CUMETAL_MSL_SIZE_LIMIT",
                         "CUMETAL_TRACE_COMPILE")}


def jit_child(runtime, source):
    """cuModuleLoad a .metal file and force preparation; print the status."""
    lib = c.CDLL(runtime)
    ptr, u32 = c.c_void_p, c.c_uint32
    lib.cuInit.argtypes, lib.cuInit.restype = [u32], c.c_int
    if lib.cuInit(0) == 100:
        return 77
    context, module, function = ptr(), ptr(), ptr()
    lib.cuCtxCreate.argtypes = [c.POINTER(ptr), u32, c.c_int]
    assert lib.cuCtxCreate(c.byref(context), 0, 0) == 0
    lib.cuModuleLoad.argtypes = [c.POINTER(ptr), c.c_char_p]
    assert lib.cuModuleLoad(c.byref(module), os.fsencode(source)) == 0
    lib.cuModuleGetFunction.argtypes = [c.POINTER(ptr), ptr, c.c_char_p]
    assert lib.cuModuleGetFunction(c.byref(function), module, b"jit_timing_probe") == 0
    value = c.c_int()
    lib.cuFuncGetAttribute.argtypes = [c.POINTER(c.c_int), c.c_int, ptr]
    status = lib.cuFuncGetAttribute(c.byref(value), 0, function)
    print("PREPARE_STATUS", status)
    return 0


def main(cumetalc, runtime):
    with tempfile.TemporaryDirectory(prefix="cumetal-compile-timing-") as work:
        work = Path(work)
        source = work / "timing_probe.cu"
        source.write_text(CUDA)
        exe = work / "timing_probe"

        # 1. Silent by default.
        result = subprocess.run([cumetalc, str(source), "-o", str(exe)], env=base_env(),
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        assert "cumetal timing:" not in result.stderr, result.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        assert run.returncode == 0 and "CUMETAL_VERSION " in run.stdout, (run.stdout, run.stderr)

        # 2. --compile-timing reports each phase, the source size and the
        #    largest function, including phases of the nested device compile.
        result = subprocess.run([cumetalc, str(source), "-o", str(exe), "--compile-timing"],
                                env=base_env(), capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        for phase in ("cuda_frontend", "nvvm_import", "metal_legalization", "msl_generation",
                      "device_compile", "host_frontend", "metal_compile", "metallib_package",
                      "host_link"):
            assert f"cumetal timing: done  {phase}" in result.stderr, (phase, result.stderr)
        assert "cumetal timing: Metal source for timing_probe.cu:" in result.stderr, result.stderr
        assert "timing_probe_kernel" in result.stderr, result.stderr
        assert "cumetal timing: summary" in result.stderr, result.stderr

        # 3. The environment variable is the same switch.
        env = dict(base_env(), CUMETAL_COMPILE_TIMING="1")
        result = subprocess.run([cumetalc, str(source), "-o", str(exe)], env=env,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0 and "cumetal timing: done  metal_compile" in result.stderr

        # 4. CUMETAL_MSL_SIZE_LIMIT fails fast and names the kernel.
        env = dict(base_env(), CUMETAL_MSL_SIZE_LIMIT="64")
        result = subprocess.run([cumetalc, str(source), "-o", str(exe), "--compile-timing"],
                                env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode != 0, "size limit did not fail the build"
        assert "over CUMETAL_MSL_SIZE_LIMIT=64" in result.stderr, result.stderr
        assert "timing_probe_kernel" in result.stderr, result.stderr
        assert "cumetal timing: done  metal_compile" not in result.stderr, \
            "Metal compiled a source over the limit"

        # 5. A malformed limit is an error, not ignored.
        env = dict(base_env(), CUMETAL_MSL_SIZE_LIMIT="huge")
        result = subprocess.run([cumetalc, str(source), "-o", str(exe)], env=env,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode != 0 and "invalid CUMETAL_MSL_SIZE_LIMIT" in result.stderr, \
            result.stderr

        # 6. Runtime path: Metal source compiled in-process.
        msl = work / "jit_probe.metal"
        msl.write_text(MSL)
        Path(str(msl) + ".cumetal-abi").write_text(
            "CUMETAL_ABI_V1\nkernel jit_timing_probe\narg buffer 8\nshared 0\n")
        child = [sys.executable, __file__, "--jit-child", runtime, str(msl)]
        env = dict(base_env(), CUMETAL_COMPILE_TIMING="1", CUMETAL_CACHE_DIR=str(work / "cache"))
        result = subprocess.run(child, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode == 77:
            print("SKIP: no supported Metal device (cumetalc checks passed)")
            return 77
        assert result.returncode == 0 and "PREPARE_STATUS 0" in result.stdout, \
            (result.stdout, result.stderr)
        for phase in ("metal_new_library_with_source", "metal_new_compute_pipeline"):
            assert f"cumetal timing: done  {phase}" in result.stderr, (phase, result.stderr)
        assert "largest functions: jit_timing_probe" in result.stderr, result.stderr

        env = dict(base_env(), CUMETAL_MSL_SIZE_LIMIT="64", CUMETAL_CACHE_DIR=str(work / "cache"))
        result = subprocess.run(child, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "PREPARE_STATUS 0" not in result.stdout, "over-limit source was prepared"
        assert "over CUMETAL_MSL_SIZE_LIMIT=64" in result.stderr, result.stderr

    print("PASS: compile timing and MSL size limit through cumetalc and the runtime")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--jit-child":
        sys.exit(jit_child(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1], sys.argv[2]))
