#!/usr/bin/env python3
"""Emit Claim Wire `@cw:record` frames for numbers that documentation quotes
from checked-in manifests and recorded results.

Run through `cw capture --producer <id>` (each producer in
`.claimwire/evidence.yaml` names its subcommand), then `cw sync`.
Manifest counting rules match tests/unit/check_documentation_consistency.sh.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def manifest_rows(rel, min_fields):
    for line in (ROOT / rel).read_text().splitlines():
        fields = line.split()
        if len(fields) >= min_fields and not fields[0].startswith("#"):
            yield fields


def record(family=None, **fields):
    parts = [f'family="{family}"'] if family else []
    for key, value in fields.items():
        parts.append(f'{key}="{value}"' if isinstance(value, str) else f"{key}={value!r}")
    print("@cw:record " + " ".join(parts))


def backend_matrix():
    rows = list(manifest_rows("tests/cuda_projects/backend_matrix_manifest.txt", 5))
    record(
        total=len(rows),
        direct_legacy=sum(r[1] == "pass" for r in rows),
        direct_typed=sum(r[2] == "pass" for r in rows),
        ptx_legacy=sum(r[3] == "pass" for r in rows),
        ptx_typed=sum(r[4] == "pass" for r in rows),
    )


def samples_sweep():
    rows = list(manifest_rows("tests/cuda_projects/cuda_samples_sweep_manifest.txt", 2))
    record(
        total=len(rows),
        **{"pass": sum(r[0] == "pass" for r in rows)},
        waive=sum(r[0] == "waive" for r in rows),
    )


def phase4_manifest():
    rows = list(manifest_rows("tests/conformance/phase4_functional_manifest.txt", 1))
    record(total=len(rows))


def coverage_manifest():
    # Projects without `backends` run on every backend (tests/cuda_projects/sweep_cuda_projects.py).
    projects = json.loads((ROOT / "tests/cuda_projects/coverage_manifest.json").read_text())["projects"]
    enrolled = lambda b: sum(b in p.get("backends", [b]) for p in projects)
    record(total=len(projects), typed_ptx=enrolled("cumetal-ir"), native_aot=enrolled("native-aot"))


MATMUL_RESULTS = "docs/evidence/matmul-2026-09-06/results.json"


def matmul():
    by_variant = {}
    for row in json.loads((ROOT / MATMUL_RESULTS).read_text()):
        by_variant.setdefault(row["name"], []).append(row)
    for name, rounds in by_variant.items():
        record(
            "matmul",
            variant=name,
            time=f"{statistics.median(r['mean_ms'] for r in rounds)}ms",
            gflops=statistics.median(r["gflops"] for r in rounds),
        )


LAMMPS_GATE = "demos/lammps/results/2026-10-04-m4-pro.json"
LAMMPS_WARM = "demos/lammps/results/2026-10-05-m4-pro-warm.json"


def hms_seconds(text):
    h, m, s = (int(x) for x in text.split(":"))
    return h * 3600 + m * 60 + s


def lammps_gate():
    d = json.loads((ROOT / LAMMPS_GATE).read_text())
    runs, reg = d["gpu_runs"], d["regressions"]
    record(
        atoms=d["atoms"],
        steps=d["steps"],
        tolerance=d["tolerance"],
        cpu_double_error=d["max_scaled_cpu_double_error"],
        mode_difference=d["max_scaled_mode_difference"],
        analytic_error=d["max_scaled_analytic_initial_error"],
        full_launches=runs["full"]["successful_apple_gpu_launches"],
        half_launches=runs["half"]["successful_apple_gpu_launches"],
        full_loop=f"{runs['full']['loop_seconds']}s",
        half_loop=f"{runs['half']['loop_seconds']}s",
        full_wall=f"{hms_seconds(runs['full']['total_wall_time'])}s",
        half_wall=f"{hms_seconds(runs['half']['total_wall_time'])}s",
        ctest_passed=reg["ctest_passed"],
        ctest_total=reg["ctest_total"],
        typed_ptx_projects=reg["typed_ptx_numerical_projects"],
        native_aot_projects=reg["native_aot_numerical_projects"],
        cuda_samples=reg["cuda_samples"],
        validator_tests=reg["demo_validator_tests"],
    )


def lammps_warm():
    d = json.loads((ROOT / LAMMPS_WARM).read_text())
    for mode in ("full", "half"):
        s = d["summary"][mode]
        record("lammps_warm", mode=mode, cpu=f"{s['cpu']}s", gpu=f"{s['gpu']}s")


PRODUCERS = {
    "backend-matrix": backend_matrix,
    "samples-sweep": samples_sweep,
    "phase4-manifest": phase4_manifest,
    "coverage-manifest": coverage_manifest,
    "matmul": matmul,
    "lammps-gate": lammps_gate,
    "lammps-warm": lammps_warm,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PRODUCERS:
        sys.exit(f"usage: {sys.argv[0]} {'|'.join(PRODUCERS)}")
    PRODUCERS[sys.argv[1]]()
