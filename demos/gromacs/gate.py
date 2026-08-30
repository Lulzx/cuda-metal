#!/usr/bin/env python3
"""Compare the per-step energy blocks of two GROMACS md.log files.

GROMACS writes energies as a two-line block: a row of column labels, then a row
of values in the same order. Labels contain spaces ("Coul. recip.", "LJ (SR)"),
so the columns are split on fixed 15-character fields rather than whitespace.

Usage:  gate.py <reference.log> <candidate.log> [--rtol R] [--atol A]
                [--envelope <reordered-reference.log>]

Exit 0 only if both logs contain the same steps and every energy term agrees.

A fixed tolerance cannot tell a defect from summation noise, because the noise
grows with system size: summing the same binary32 forces in a different order
moves ADH's total energy by 1.3e-3 relative between two thread counts of the
*same* CPU build, six times the tolerance that villin comfortably meets. Pass
--envelope with a second reference run that differs only in something physically
irrelevant (thread count) and each term's tolerance is widened to the difference
that run shows, so the bar is "closer to the reference than the reference is to
itself" rather than a number guessed per system.
"""
import argparse
import re
import sys

FIELD = 15
# Terms compared. Pressure is excluded: it is a virial estimate that swings by
# hundreds of bar between neighbouring steps in a 5000-atom box, so agreeing on
# it says nothing that the energies have not already said. Constr. rmsd is a
# solver residual, not a physical quantity.
SKIP = {"Pressure (bar)", "Constr. rmsd", "Time"}


def split_row(line):
    """Split a fixed-width GROMACS energy row into stripped fields."""
    out = []
    for i in range(0, len(line.rstrip("\n")), FIELD):
        cell = line[i : i + FIELD].strip()
        if cell:
            out.append(cell)
    return out


def parse(path):
    """Return {step: {term: value}} for every energy block in a GROMACS log."""
    blocks = {}
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "Step           Time":
            step = int(lines[i + 1].split()[0])
            j = i + 2
            while j < len(lines) and "Energies (kJ/mol)" not in lines[j]:
                # A step header is always followed by its energy block within a
                # few lines; anything longer means this was a different table.
                if j - i > 6:
                    break
                j += 1
            if j < len(lines) and "Energies (kJ/mol)" in lines[j]:
                terms = {}
                k = j + 1
                while k + 1 < len(lines) and lines[k].strip():
                    labels = split_row(lines[k])
                    values = split_row(lines[k + 1])
                    if len(labels) != len(values):
                        break
                    try:
                        for lab, val in zip(labels, values):
                            terms[lab] = float(val)
                    except ValueError:
                        break
                    k += 2
                if terms:
                    blocks[step] = terms
                i = k
                continue
        i += 1
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--label", default="")
    ap.add_argument("--envelope", action="append", default=[],
                    help="reference run differing only in thread count; repeatable")
    ap.add_argument("--envelope-factor", type=float, default=1.0)
    args = ap.parse_args()

    ref = parse(args.reference)
    cand = parse(args.candidate)
    # One reordering is not an estimate of the noise: on a 192k-atom case the
    # same reference lands 5.2e-05 away at two threads and 1.3e-03 away at one.
    # Take the widest of several, so the floor is a bound rather than a sample.
    envs = []
    for path in args.envelope:
        e = parse(path)
        if not e:
            print(f"FAIL: no energy blocks in {path}")
            return 1
        envs.append(e)
    env = {}
    for e in envs:
        for st, terms in e.items():
            for tm, v in terms.items():
                base = ref.get(st, {}).get(tm)
                if base is None:
                    continue
                if abs(v - base) > abs(env.get(st, {}).get(tm, base) - base):
                    env.setdefault(st, {})[tm] = v

    if not ref:
        print(f"FAIL: no energy blocks in {args.reference}")
        return 1
    if not cand:
        print(f"FAIL: no energy blocks in {args.candidate}")
        return 1

    steps = sorted(set(ref) & set(cand))
    if not steps:
        print(f"FAIL: no common steps (ref {sorted(ref)[:5]}, cand {sorted(cand)[:5]})")
        return 1

    worst = []
    fails = 0
    for step in steps:
        for term, rv in ref[step].items():
            if term in SKIP or term not in cand[step]:
                continue
            cv = cand[step][term]
            if cv != cv:  # NaN
                print(f"FAIL: step {step} {term}: candidate is NaN")
                fails += 1
                continue
            diff = abs(cv - rv)
            tol = args.atol + args.rtol * abs(rv)
            # The reference's disagreement with itself is a floor on what any
            # correct reimplementation can achieve, not slack to be explained.
            noise = abs(env.get(step, {}).get(term, rv) - rv) if env else 0.0
            tol = max(tol, args.envelope_factor * noise)
            rel = diff / abs(rv) if rv else diff
            worst.append((rel, step, term, rv, cv))
            if diff > tol:
                print(
                    f"FAIL: step {step:>4} {term:<15} ref={rv:.6e} "
                    f"got={cv:.6e} rel={rel:.2e} (tol {args.rtol:.0e})"
                )
                fails += 1

    worst.sort(reverse=True)
    prefix = f"{args.label}: " if args.label else ""
    note = ""
    if env:
        floors = [
            abs(env[st][tm] - ref[st][tm]) / abs(ref[st][tm])
            for st in steps
            for tm in ref[st]
            if tm not in SKIP and tm in env.get(st, {}) and ref[st][tm]
        ]
        note = f", reference-vs-itself noise floor {max(floors, default=0.0):.2e}"
    print(f"{prefix}{len(steps)} steps compared, {len(worst)} term comparisons{note}")
    for rel, step, term, rv, cv in worst[:3]:
        print(f"  largest: step {step:>4} {term:<15} rel={rel:.2e} ref={rv:.6e} got={cv:.6e}")
    if fails:
        print(f"{prefix}FAIL: {fails} term(s) outside tolerance")
        return 1
    print(f"{prefix}PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
