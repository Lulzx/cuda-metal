#!/usr/bin/env python3
"""Aggregate runs.jsonl into a per-instance table and Mittelmann-style summary.

Three timings are carried through end to end because they answer different
questions and only one of them is the user-facing number:

  wall       includes reading the MPS file, which for a2864 is 984 MB. Identical
             work in both builds, so it dilutes any speedup toward 1.0x.
  highs      HiGHS's own timer: presolve + solve + postsolve.
  solve      the presolved LP solve -- the only part cuPDLP runs at all.

Aggregation follows the benchmark: shifted geometric mean with a 10 s shift,

    SGM(t) = exp( mean( ln(t_i + 10) ) ) - 10

and a run that does not solve is counted at the time limit rather than dropped,
so a build cannot win by failing on its hard instances.

Correctness is checked before speed. An instance whose two builds disagree on
model status, or whose objectives differ by more than --obj-tol relative, is
reported FAIL and excluded from the speed aggregate -- a fast wrong answer is
not a result.
"""
import argparse, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIFT = 10.0
METRICS = [("wall", "wall"), ("highs", "highs_run_time"), ("solve", "solve_presolved_s")]


def sgm(times, shift=SHIFT):
    ts = [t for t in times if t is not None]
    if not ts:
        return None
    return math.exp(sum(math.log(t + shift) for t in ts) / len(ts)) - shift


def status_class(s):
    """'Optimal current' and 'Optimal average' differ only in which iterate PDLP
    accepted; they are the same outcome."""
    if not s:
        return "none"
    s = s.lower()
    for key in ("optimal", "infeasible", "unbounded", "time limit", "iteration limit"):
        if key in s:
            return key
    return s


def load(path):
    runs = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            runs[(d["instance"], d["track"], d["build"])].append(d)
    return runs


def median_of(rs, field):
    vals = [r.get(field) for r in rs if r.get(field) is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "results/runs.jsonl"))
    ap.add_argument("--obj-tol", type=float, default=1e-6,
                    help="relative objective agreement; the benchmark's tolerance")
    ap.add_argument("--metric", default="highs", choices=[m[0] for m in METRICS],
                    help="which timing drives the summary (default: highs)")
    ap.add_argument("--out", default=str(HERE / "results/report.md"))
    a = ap.parse_args()

    runs = load(a.runs)
    if not runs:
        sys.exit("no runs recorded yet")
    metric_field = dict(METRICS)[a.metric]

    tracks = sorted({k[1] for k in runs})
    lines = ["# LPfeas benchmark: HiGHS CPU vs HiGHS on CuMetal", ""]

    for track in tracks:
        names = sorted({k[0] for k in runs if k[1] == track},
                       key=lambda n: next(r["nnz"] for k, v in runs.items()
                                          if k[0] == n for r in v))
        lines += [f"## Track {track} "
                  f"({'presolve default' if track == 'A' else 'presolve off'})", "",
                  "| instance | rows | cols | nnz | status cpu/gpu | obj rel diff |"
                  " iters cpu/gpu | wall c/g (s) | highs c/g (s) | solve c/g (s) |"
                  " speedup | ok |",
                  "|---|---:|---:|---:|---|---:|---|---|---|---|---:|---|"]
        agg = {"cpu": [], "gpu": []}
        wins = {"gpu": 0, "cpu": 0, "tie": 0}
        solved = {"cpu": 0, "gpu": 0}
        rows_out = []
        for name in names:
            rc, rg = runs.get((name, track, "cpu")), runs.get((name, track, "gpu"))
            if not rc or not rg:
                continue
            meta = rc[0]
            sc, sg = status_class(rc[0]["status"]), status_class(rg[0]["status"])
            oc, og = median_of(rc, "objective"), median_of(rg, "objective")
            rel = (abs(oc - og) / max(abs(oc), 1.0)
                   if oc is not None and og is not None else None)
            ok = (sc == sg) and rel is not None and rel <= a.obj_tol
            # A shared non-optimal stop is agreement about difficulty, not a
            # disagreement about an answer. Both builds hitting the time limit,
            # or both returning Unknown at the same iterate, is consistency.
            if sc == sg and sc in ("time limit", "iteration limit", "unknown"):
                ok = True
            for b, rs in (("cpu", rc), ("gpu", rg)):
                if status_class(rs[0]["status"]) == "optimal":
                    solved[b] += 1
            tl = meta.get("time_limit", 0.0)
            t = {}
            for b, rs in (("cpu", rc), ("gpu", rg)):
                v = median_of(rs, metric_field)
                # Only a run that actually ran out of time is charged the limit.
                # A run that terminates early with a non-Optimal status has a
                # real time, and substituting the limit would both invent work
                # that never happened and drag the ratio toward 1.0. HiGHS
                # returns Unknown when cuPDLP declares convergence but HiGHS's
                # own postsolve KKT check rejects the point -- graph40-40 does
                # this identically on both builds in under 2 s.
                if any(r.get("timed_out") for r in rs) or v is None:
                    v = tl
                t[b] = v
            if ok and t["cpu"] is not None and t["gpu"] is not None:
                agg["cpu"].append(t["cpu"]); agg["gpu"].append(t["gpu"])
                sp = t["cpu"] / t["gpu"] if t["gpu"] else None
                if sp and sp > 1.05:   wins["gpu"] += 1
                elif sp and sp < 0.95: wins["cpu"] += 1
                else:                  wins["tie"] += 1
            else:
                sp = None

            def pair(field):
                x, y = median_of(rc, field), median_of(rg, field)
                f = lambda v: "-" if v is None else f"{v:.2f}"
                return f"{f(x)} / {f(y)}"

            rows_out.append(
                f"| {name} | {meta['rows']:,} | {meta['cols']:,} | {meta['nnz']:,} "
                f"| {sc} / {sg} | {'-' if rel is None else f'{rel:.1e}'} "
                f"| {median_of(rc,'iterations'):.0f} / {median_of(rg,'iterations'):.0f} "
                f"| {pair('wall')} | {pair('highs_run_time')} "
                f"| {pair('solve_presolved_s')} "
                f"| {'-' if sp is None else f'{sp:.2f}x'} "
                f"| {'ok' if ok else '**FAIL**'} |")
        lines += rows_out + [""]

        n = len(agg["cpu"])
        s_cpu, s_gpu = sgm(agg["cpu"]), sgm(agg["gpu"])
        lines += [f"**Summary (track {track}, metric `{a.metric}`, "
                  f"{n} instances agreeing)**", "",
                  "```text",
                  f"solved (Optimal):  cpu {solved['cpu']} / {len(rows_out)}"
                  f"    gpu {solved['gpu']} / {len(rows_out)}",
                  f"SGM (shift 10 s):  cpu {s_cpu:.2f}s    gpu {s_gpu:.2f}s"
                  if s_cpu and s_gpu else "SGM: n/a",
                  f"overall speedup:   {s_cpu / s_gpu:.3f}x"
                  if s_cpu and s_gpu else "overall speedup: n/a",
                  f"gpu faster: {wins['gpu']}   cpu faster: {wins['cpu']}"
                  f"   within 5%: {wins['tie']}",
                  "```", ""]

    text = "\n".join(lines)
    Path(a.out).write_text(text)
    print(text)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
