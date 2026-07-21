#!/usr/bin/env python3
import csv
import math
import sys


def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    if len(sys.argv) != 7:
        print(
            "usage: compare_physx_grb_multibody.py "
            "CPU.tsv GPU.tsv STEPS BODIES REL_TOL ABS_TOL",
            file=sys.stderr,
        )
        return 2

    cpu_path, gpu_path = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3])
    bodies = int(sys.argv[4])
    rel_tol = float(sys.argv[5])
    abs_tol = float(sys.argv[6])
    cpu_rows = load(cpu_path)
    gpu_rows = load(gpu_path)
    expected_rows = (steps + 1) * bodies
    if len(cpu_rows) != expected_rows or len(gpu_rows) != expected_rows:
        print(
            f"FAIL: expected {expected_rows} body-state rows; "
            f"CPU={len(cpu_rows)} GPU={len(gpu_rows)}"
        )
        return 1

    fields = (
        "px", "py", "pz", "qx", "qy", "qz", "qw",
        "vx", "vy", "vz", "wx", "wy", "wz",
    )
    worst = (0.0, 0, 0, "")
    for index, (cpu, gpu) in enumerate(zip(cpu_rows, gpu_rows)):
        expected_step = index // bodies
        expected_body = index % bodies
        identity = (str(expected_step), str(expected_body))
        if (cpu["step"], cpu["body"]) != identity or (
            gpu["step"], gpu["body"]
        ) != identity:
            print(f"FAIL: row identity mismatch at row {index}")
            return 1
        for field in fields:
            reference = float(cpu[field])
            actual = float(gpu[field])
            if not math.isfinite(reference) or not math.isfinite(actual):
                print(
                    f"FAIL: non-finite {field} at step={expected_step} "
                    f"body={expected_body}"
                )
                return 1
            error = abs(actual - reference)
            limit = abs_tol + rel_tol * max(abs(reference), 1.0)
            ratio = error / limit if limit else error
            if ratio > worst[0]:
                worst = (ratio, expected_step, expected_body, field)
            if error > limit:
                print(
                    f"FAIL: step={expected_step} body={expected_body} "
                    f"field={field} CPU={reference:.9g} GPU={actual:.9g} "
                    f"error={error:.3g} tolerance={limit:.3g}"
                )
                return 1

    print(
        f"PASS: PhysX CPU/GPU states match for {bodies} bodies over "
        f"{steps} steps (rel={rel_tol:g}, abs={abs_tol:g}; "
        f"worst={worst[3] or 'none'}@{worst[1]}/body{worst[2]} "
        f"ratio={worst[0]:.3g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
