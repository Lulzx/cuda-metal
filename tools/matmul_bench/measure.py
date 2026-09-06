#!/usr/bin/env python3
"""Three 20-iteration rounds in rotated order; preserve every check and timing."""
import argparse
import json
import pathlib
import subprocess

p = argparse.ArgumentParser()
p.add_argument("build", type=pathlib.Path)
p.add_argument("--output", type=pathlib.Path, required=True)
p.add_argument("--iterations", type=int, default=20)
p.add_argument("--size", type=int, default=4096)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
variants = ["mps", "naive", "coalesced", "tiled", "registers", "vectorized", "block64_32", "cublas_mps", "accelerate"]
results = []
for repeat in range(3):
    order = variants[repeat * 3:] + variants[:repeat * 3]
    for name in order:
        command = [str(a.build / "matmul_bench"), str(a.build / "kernels.metallib"), *([str(a.size)] * 3), str(a.iterations), name]
        run = subprocess.run(command, capture_output=True, text=True)
        (a.output / f"round{repeat + 1}-{name}.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise SystemExit(f"Failed {name}: {run.stdout}\n{run.stderr}")
        for line in run.stdout.splitlines():
            if line.startswith("result,"):
                values = line.split(",")
                results.append(dict(round=repeat + 1, name=name, mean_ms=float(values[6]), median_ms=float(values[7]), min_ms=float(values[8]), max_ms=float(values[9]), gflops=float(values[10])))
                print(line, flush=True)
        (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
