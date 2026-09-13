#!/usr/bin/env python3
"""Translate unchanged Rust-generated PTX and require numerical Apple-GPU evidence."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import shlex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["vecadd", "sha256", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    ptx = args.ptx.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUMETAL_TRACE_GPU="1", CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="0")
    compiler = build / "cumetalc"
    runner = out / "rust-ptx-runner"
    subprocess.run(["/usr/bin/clang++", "-std=c++20", "-Wall", "-Wextra",
                    "-I", str(root / "runtime/api"), str(Path(__file__).parent / "runner.cpp"),
                    "-L", str(build), "-lcumetal", f"-Wl,-rpath,{build}", "-o", str(runner)], check=True)
    digest = hashlib.sha256(ptx.read_bytes()).hexdigest()
    (out / "ptx-sha256.txt").write_text(f"{digest}  {ptx}\n")
    modes = ["vecadd", "sha256"] if args.mode == "all" else [args.mode]
    for mode in modes:
        kernel = "rust_vecadd" if mode == "vecadd" else "rust_sha256_32"
        metal = out / f"{kernel}.metal"
        # Explicit typed backend; unsupported PTX is a failure, with no fallback.
        command = [str(compiler), str(ptx), "--backend=cumetal-ir", "--ptx-strict", "--overwrite",
                   "--entry", kernel, "--emit=msl", "-o", str(metal)]
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / f"{mode}-compile.log").write_text(result.stdout)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
        if not Path(str(metal) + ".cumetal-abi").is_file():
            raise RuntimeError("compiler did not emit argument metadata")
        result = subprocess.run([str(runner), str(metal), mode], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / f"{mode}-run.log").write_text(result.stdout)
        print(result.stdout, end="", flush=True)
        result.check_returncode()
        for count in (1, 31, 32, 33, 257):
            assert f"NUMERICAL_PASS kernel={kernel} count={count}" in result.stdout
        records = []
        for line in result.stdout.splitlines():
            if line.startswith("CUMETAL_PROVENANCE "):
                fields = dict(word.split("=", 1) for word in shlex.split(line)[1:] if "=" in word)
                if fields.get("kernel") == kernel:
                    records.append(fields)
        if len(records) != 5 or not all(record.get("device") == "apple_gpu" and
                record.get("launch_success") == "true" and
                record.get("provenance") == "generic_ptx_lowering" for record in records):
            raise RuntimeError("missing generic PTX Apple-GPU execution evidence")
    print("PASS: requested PTX kernels executed numerically correctly on Apple GPU")


if __name__ == "__main__":
    main()
