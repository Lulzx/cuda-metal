"""Validate completion and finite output of the stock CPU benchmark."""
import json
import math
from pathlib import Path
import sys


def read_reference(path, mode, precision="single"):
    text = path.read_text()
    for expected in (
        "LAMMPS (30 Sep 2026)",
        "Kokkos version 5.2.1",
        f"using {precision} precision",
        "Created 32000 atoms",
        f"attributes: {mode}, newton {'off' if mode == 'full' else 'on'}",
        "for 100 steps with 32000 atoms",
    ):
        if expected not in text:
            raise ValueError(f"{path}: missing {expected!r}")
    if "ERROR" in text:
        raise ValueError(f"{path}: LAMMPS error")
    rows = {}
    in_thermo = False
    for line in text.splitlines():
        if line.split() == ["Step", "Temp", "E_pair", "E_mol", "TotEng", "Press"]:
            in_thermo = True
            continue
        if line.startswith("Loop time"):
            in_thermo = False
        if not in_thermo:
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}: malformed thermo row {line!r}")
        values = [float(v) for v in fields]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}: nonfinite thermo output")
        step = int(values[0])
        if values[0] != step or step in rows:
            raise ValueError(f"{path}: invalid or duplicate timestep")
        rows[step] = values[1:]
    if set(rows) != {0, 100}:
        raise ValueError(f"{path}: incomplete trajectory")
    if rows[0][0] != 1.44 or rows[100][0] <= 0 or rows[100][1] >= 0:
        raise ValueError(f"{path}: degenerate Lennard-Jones output")
    return rows


def main():
    output = Path(sys.argv[1])
    precision = sys.argv[2] if len(sys.argv) > 2 else "single"
    modes = {m: read_reference(output / f"{m}.log", m, precision) for m in ("full", "half")}
    difference = max(
        abs(a - b) / max(1.0, abs(a))
        for step in (0, 100)
        for a, b in zip(modes["full"][step], modes["half"][step])
    )
    result = {
        "backend": "kokkos_serial_cpu",
        "precision": precision,
        "lammps_revision": "8de817dd79bfe4525d5d39246a212d833e6dee07",
        "atoms": 32000,
        "steps": 100,
        "thermo_fields": ["temperature", "pair_energy", "molecular_energy", "total_energy", "pressure"],
        "thermo": modes,
        "max_scaled_mode_difference": difference,
        "agreement_tolerance": 5e-4,
        "agreement_status": "fail" if difference > 5e-4 else "pass",
        "gpu_validation": "not_performed",
    }
    (output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    if difference > 5e-4:
        raise ValueError(f"CPU neighbour modes disagree: {difference:.6g}; see reference.json")
    print(f"CPU REFERENCE PASS: 32000 atoms, 100 steps, mode difference={difference:.6g}")


if __name__ == "__main__":
    main()
