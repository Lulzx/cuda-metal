#!/usr/bin/env python3
"""Bundled-cache contract (#229): with CUMETAL_CACHE_DIR unset, a relocated
libcumetal.dylib uses <its directory>/../share/cumetal-cache when that exists
and is writable; a read-only one falls back to the per-user cache with a
warning; CUMETAL_CACHE_DIR always wins.

Each case runs in a child process so the cache root is resolved fresh, loads
the relocated dylib with ctypes, and stages a metallib through
cuModuleLoadData, which writes it into <cache root>/kernels/."""
import ctypes as c
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void bundled_cache_probe(device uint* out [[buffer(0)]],
                                uint i [[thread_position_in_grid]]) { out[i] = i; }
"""


def child(dylib, metallib):
    lib = c.CDLL(dylib)
    ptr, u32 = c.c_void_p, c.c_uint32
    lib.cuInit.argtypes, lib.cuInit.restype = [u32], c.c_int
    status = lib.cuInit(0)
    if status == 100:
        return 77
    assert status == 0, ("cuInit", status)
    context, module = ptr(), ptr()
    lib.cuCtxCreate.argtypes = [c.POINTER(ptr), u32, c.c_int]
    assert lib.cuCtxCreate(c.byref(context), 0, 0) == 0
    image = Path(metallib).read_bytes()
    buffer = c.create_string_buffer(image, len(image))
    lib.cuModuleLoadData.argtypes = [c.POINTER(ptr), ptr]
    status = lib.cuModuleLoadData(c.byref(module), c.cast(buffer, ptr))
    assert status == 0, ("cuModuleLoadData", status)
    lib.cuModuleUnload.argtypes = [ptr]
    lib.cuModuleUnload(module)
    lib.cuCtxDestroy.argtypes = [ptr]
    lib.cuCtxDestroy(context)
    return 0


def kernels(root):
    directory = Path(root) / "kernels"
    return sorted(p.name for p in directory.glob("*")) if directory.is_dir() else []


def run_case(dylib, metallib, env):
    result = subprocess.run([sys.executable, __file__, "--child", dylib, metallib],
                            env=env, capture_output=True, text=True, timeout=120)
    return result


def main(runtime):
    if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode != 0:
        print("SKIP: no Metal toolchain to build the probe metallib")
        return 77
    with tempfile.TemporaryDirectory(prefix="cumetal-bundled-cache-") as work:
        work = Path(work)
        (work / "probe.metal").write_text(SOURCE)
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(work / "probe.metal"),
                        "-o", str(work / "probe.air")], check=True)
        subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(work / "probe.air"),
                        "-o", str(work / "probe.metallib")], check=True)
        metallib = str(work / "probe.metallib")

        # A wheel/tarball-shaped bundle: <prefix>/lib/libcumetal.dylib and
        # <prefix>/share/cumetal-cache.
        prefix = work / "bundle"
        (prefix / "lib").mkdir(parents=True)
        dylib = prefix / "lib" / "libcumetal.dylib"
        shutil.copy2(runtime, dylib)
        bundled = prefix / "share" / "cumetal-cache"
        bundled.mkdir(parents=True)
        home = work / "home"
        home.mkdir()
        user_cache = home / "Library" / "Caches" / "io.cumetal"

        env = {k: v for k, v in os.environ.items() if k != "CUMETAL_CACHE_DIR"}
        env["HOME"] = str(home)

        # 1. Writable bundled cache is used.
        result = run_case(str(dylib), metallib, env)
        if result.returncode == 77:
            print("SKIP: no supported Metal device")
            return 77
        assert result.returncode == 0, result.stderr
        assert kernels(bundled), ("bundled cache unused", result.stderr)
        assert not kernels(user_cache), "per-user cache written despite a bundled cache"

        # 2. CUMETAL_CACHE_DIR wins over the bundled cache.
        explicit = work / "explicit"
        before = kernels(bundled)
        result = run_case(str(dylib), metallib, dict(env, CUMETAL_CACHE_DIR=str(explicit)))
        assert result.returncode == 0, result.stderr
        # This cache has always used CUMETAL_CACHE_DIR itself as the kernels dir.
        assert any(explicit.glob("*")), ("CUMETAL_CACHE_DIR unused", result.stderr)
        assert kernels(bundled) == before, "bundled cache written despite CUMETAL_CACHE_DIR"

        # 3. A read-only bundled cache is skipped with a warning.
        shutil.rmtree(bundled)
        bundled.mkdir()
        bundled.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            result = run_case(str(dylib), metallib, env)
        finally:
            bundled.chmod(stat.S_IRWXU)
        assert result.returncode == 0, result.stderr
        assert "is not writable" in result.stderr, result.stderr
        assert kernels(user_cache), ("per-user fallback unused", result.stderr)
        assert not any(bundled.glob("*")), "read-only bundled cache was written"

        # 4. No bundled directory: the per-user cache, as before.
        shutil.rmtree(prefix / "share")
        shutil.rmtree(user_cache)
        result = run_case(str(dylib), metallib, env)
        assert result.returncode == 0, result.stderr
        assert kernels(user_cache), ("per-user cache unused", result.stderr)
        assert "is not writable" not in result.stderr

    print("PASS: bundled share/cumetal-cache next to libcumetal.dylib")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        sys.exit(child(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1]))
