#!/usr/bin/env python3
import csv
import math
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows


def main():
    if len(sys.argv) != 6:
        print(
            "usage: compare_physx_grb.py CPU.tsv GPU.tsv STEPS REL_TOL ABS_TOL",
            file=sys.stderr,
        )
        return 2

    cpu_path, gpu_path = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3])
    rel_tol = float(sys.argv[4])
    abs_tol = float(sys.argv[5])
    cpu_rows = load(cpu_path)
    gpu_rows = load(gpu_path)
    expected_rows = steps + 1
    if len(cpu_rows) != expected_rows or len(gpu_rows) != expected_rows:
        print(
            f"FAIL: expected {expected_rows} transform rows; "
            f"CPU={len(cpu_rows)} GPU={len(gpu_rows)}"
        )
        return 1

    fields = ("px", "py", "pz", "qx", "qy", "qz", "qw")
    worst = (0.0, 0, "")
    for index, (cpu, gpu) in enumerate(zip(cpu_rows, gpu_rows)):
        if cpu["step"] != gpu["step"] or int(cpu["step"]) != index:
            print(f"FAIL: step mismatch at row {index}")
            return 1
        for field in fields:
            reference = float(cpu[field])
            actual = float(gpu[field])
            if not math.isfinite(reference) or not math.isfinite(actual):
                print(f"FAIL: non-finite {field} at step {index}")
                return 1
            error = abs(actual - reference)
            limit = abs_tol + rel_tol * max(abs(reference), 1.0)
            ratio = error / limit if limit else error
            if ratio > worst[0]:
                worst = (ratio, index, field)
            if error > limit:
                print(
                    f"FAIL: step={index} field={field} CPU={reference:.9g} "
                    f"GPU={actual:.9g} error={error:.3g} tolerance={limit:.3g}"
                )
                return 1

    print(
        f"PASS: PhysX CPU/GPU transforms match for {steps} steps "
        f"(rel={rel_tol:g}, abs={abs_tol:g}; "
        f"worst={worst[2] or 'none'}@{worst[1]} ratio={worst[0]:.3g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
