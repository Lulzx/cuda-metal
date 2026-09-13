#!/usr/bin/env python3
"""Run one miner self-test from compiled MSL; require its slot=1 and intact guards."""
import argparse
import ctypes as c
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--result-count", type=int, default=118)
    args = parser.parse_args()
    if not 0 <= args.slot < args.result_count <= 4096:
        parser.error("require 0 <= slot < result-count <= 4096")
    os.environ["CUMETAL_TRACE_GPU"] = "1"
    os.environ["CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS"] = "0"
    lib = c.CDLL(str(args.build_dir.resolve() / "libcumetal.dylib"))

    def api(name, types, *values):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        code = fn(*values)
        if code:
            raise RuntimeError(f"{name} failed: {code}")

    ptr, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
    context, module, function = ptr(), ptr(), ptr()
    allocation = u64()
    poison = 0xa5a5a5a5
    words = (u32 * (args.result_count + 16))(*([poison] * (args.result_count + 16)))
    api("cuInit", [u32], 0)
    api("cuCtxCreate", [c.POINTER(ptr), u32, c.c_int], c.byref(context), 0, 0)
    try:
        api("cuModuleLoad", [c.POINTER(ptr), c.c_char_p], c.byref(module), os.fsencode(args.module.resolve()))
        api("cuModuleGetFunction", [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, args.kernel.encode())
        api("cuMemAlloc", [c.POINTER(u64), c.c_size_t], c.byref(allocation), c.sizeof(words))
        api("cuMemcpyHtoD", [u64, ptr, c.c_size_t], allocation, c.cast(words, ptr), c.sizeof(words))
        parameters = (ptr * 2)(c.cast(c.pointer(allocation), ptr), None)
        api("cuLaunchKernel", [ptr] + [u32]*7 + [ptr, c.POINTER(ptr), ptr],
            function, 1, 1, 1, 1, 1, 1, 0, None, parameters, None)
        api("cuCtxSynchronize", [])
        api("cuMemcpyDtoH", [ptr, u64, c.c_size_t], c.cast(words, ptr), allocation, c.sizeof(words))
        changed = [i for i, value in enumerate(words) if i != args.slot and value != poison]
        if changed:
            raise RuntimeError(f"unexpected writes outside result slot: {changed}")
        passed = words[args.slot] == 1
        print(f"NUMERICAL_{'PASS' if passed else 'FAIL'} kernel={args.kernel} "
              f"slot={args.slot} actual={words[args.slot]} expected=1; "
              f"other {args.result_count - 1} slots and 16 guard words intact")
        return 0 if passed else 1
    finally:
        if allocation.value:
            api("cuMemFree", [u64], allocation)
        if module.value:
            api("cuModuleUnload", [ptr], module)
        api("cuCtxDestroy", [ptr], context)


if __name__ == "__main__":
    raise SystemExit(main())
