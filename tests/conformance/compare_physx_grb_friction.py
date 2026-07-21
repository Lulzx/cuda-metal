#!/usr/bin/env python3
import csv
import math
import sys


FIELDS = ("px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")


def load(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def fail(message):
    print(f"FAIL: {message}")
    return 1


def value(row, field):
    result = float(row[field])
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def main():
    if len(sys.argv) != 6:
        print(
            "usage: compare_physx_grb_friction.py CPU_FRICTION.tsv "
            "GPU_FRICTION.tsv GPU_FRICTIONLESS.tsv STEPS EARLY_STEPS",
            file=sys.stderr,
        )
        return 2

    cpu = load(sys.argv[1])
    gpu = load(sys.argv[2])
    frictionless = load(sys.argv[3])
    steps = int(sys.argv[4])
    early_steps = int(sys.argv[5])
    expected = steps + 1
    if any(len(rows) != expected for rows in (cpu, gpu, frictionless)):
        return fail(
            f"expected {expected} rows; CPU={len(cpu)} GPU={len(gpu)} "
            f"frictionless={len(frictionless)}"
        )

    try:
        for index, rows in enumerate(zip(cpu, gpu, frictionless)):
            if any(int(row["step"]) != index for row in rows):
                return fail(f"step mismatch at row {index}")
            for row in rows:
                for field in FIELDS:
                    value(row, field)

        for index in range(min(early_steps, steps) + 1):
            for field in FIELDS:
                reference = value(cpu[index], field)
                actual = value(gpu[index], field)
                limit = 5e-4 + 1e-3 * max(abs(reference), 1.0)
                if abs(actual - reference) > limit:
                    return fail(
                        f"kinetic phase mismatch step={index} field={field} "
                        f"CPU={reference:.9g} GPU={actual:.9g}"
                    )

        cpu_final = cpu[-1]
        gpu_final = gpu[-1]
        off_final = frictionless[-1]
        cpu_vx = value(cpu_final, "vx")
        gpu_vx = value(gpu_final, "vx")
        off_vx = value(off_final, "vx")
        cpu_wz = value(cpu_final, "wz")
        gpu_wz = value(gpu_final, "wz")
        gpu_x = value(gpu_final, "px")
        off_x = value(off_final, "px")

        if not (2.0 < cpu_vx < 4.0 and cpu_wz < -2.0):
            return fail(f"CPU friction reference did not reach rolling state: vx={cpu_vx:g} wz={cpu_wz:g}")
        if abs(gpu_vx - cpu_vx) > 0.01 or abs(gpu_wz - cpu_wz) > 0.01:
            return fail(
                f"GPU rolling state mismatch: CPU vx={cpu_vx:g} wz={cpu_wz:g}; "
                f"GPU vx={gpu_vx:g} wz={gpu_wz:g}"
            )
        if abs(gpu_vx + gpu_wz) > 0.01:
            return fail(f"GPU did not reach no-slip rolling: vx+wz={gpu_vx + gpu_wz:g}")
        if abs(off_vx - 5.0) > 0.02 or abs(value(off_final, "wz")) > 0.02:
            return fail(f"friction-disabled control changed tangential motion: vx={off_vx:g}")
        if off_x - gpu_x < 1.0:
            return fail(f"friction did not materially reduce travel: friction={gpu_x:g} disabled={off_x:g}")
    except (KeyError, ValueError) as error:
        return fail(str(error))

    print(
        "PASS: PhysX sphere/plane friction matches CPU sliding and rolling; "
        f"GPU step {steps} vx={gpu_vx:.6g}, wz={gpu_wz:.6g}, "
        f"disabled vx={off_vx:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
