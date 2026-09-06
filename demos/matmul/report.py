#!/usr/bin/env python3
"""Summarize the demo's three-round measurement without a speed threshold."""
import json
import math
import pathlib
import statistics
import sys

rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
size = int(sys.argv[2])
names = ["naive", "coalesced", "tiled", "registers", "vectorized", "block64_32", "accelerate", "cublas_mps", "mps"]
if len(rows) != 3 * len(names):
    raise SystemExit("FAIL: incomplete measurement set")
means = {}
for name in names:
    group = [row for row in rows if row["name"] == name]
    if len(group) != 3 or {row["round"] for row in group} != {1, 2, 3}:
        raise SystemExit(f"FAIL: missing/duplicate rounds for {name}")
    values = [row["mean_ms"] for row in group]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise SystemExit(f"FAIL: invalid timing for {name}")
    means[name] = statistics.median(values)
print(f"# CUDA matmul vs MPS: {size} x {size} x {size} FP32\n")
print("Median of three round means; synchronized wall time, two warmups per round.\n")
if size != 4096:
    print("Quick smoke run: these timings do not reproduce the 4096-cubed study.\n")
print("| Kernel / library | ms | GFLOP/s | % of MPS |")
print("|---|---:|---:|---:|")
for name, ms in means.items():
    print(f"| {name} | {ms:.3f} | {2 * size**3 / (ms * 1e6):.1f} | {100 * means['mps'] / ms:.1f}% |")
print(f"\nTuned CUDA speedup over article vectorized: {means['vectorized'] / means['block64_32']:.2f}x.")
print("\n`cublas_mps` is the existing explicit CuMetal library call to MPS; `block64_32` is compiled CUDA.")
