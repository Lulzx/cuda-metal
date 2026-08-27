#!/usr/bin/env python3
"""Run the LPfeas benchmark: HiGHS CPU build vs HiGHS CuMetal GPU build.

Two tracks, per Mittelmann's setup but on one laptop:

  A  --solver pdlp, presolve default   the user-facing question: does installing
                                       CuMetal make HiGHS faster end to end?
  B  --solver pdlp, --presolve off     the engineering question: is the PDLP
                                       solve loop itself faster on the GPU?

Both builds come from one source tree and differ only in -DCUPDLP_GPU
(scripts/build_highs_cumetal.sh), so a difference between them is the GPU path.

Three timings are recorded per run and they are not interchangeable:

  wall              external, includes reading the MPS file. a2864's MPS is
                    984 MB, so for a fast solve this is mostly I/O and parsing,
                    which is identical in both builds and dilutes any speedup.
  highs_run_time    HiGHS's own timer: presolve + solve + postsolve, no read.
  solve_presolved   the part cuPDLP actually runs. On track A the rest is
                    presolve, which is CPU-only in both builds by construction.

Reporting only the first would repeat the datt256 mistake, where presolve
dominated the wall clock and hid what the GPU did to the solve loop.

Results append to results/runs.jsonl, one JSON object per run, so an
interrupted sweep resumes instead of starting over.
"""
import argparse, json, os, re, resource, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch import instances  # noqa: E402

TRACKS = {
    "A": {"presolve": "on"},   # HiGHS's default for pdlp is to presolve
    "B": {"presolve": "off"},
}

# HiGHS prints its phase breakdown only at log_dev_level=1. The overhead is a
# handful of extra lines and applies to both builds equally.
_RE = {
    "status":      re.compile(r"^Model status\s*:\s*(.+?)\s*$", re.M),
    "objective":   re.compile(r"^Objective value\s*:\s*(\S+)", re.M),
    "iterations":  re.compile(r"^PDLP\s+iterations:\s*(\d+)", re.M),
    "highs_time":  re.compile(r"^HiGHS run time\s*:\s*(\S+)", re.M),
    "pinf":        re.compile(r"Primal infeas \(abs/rel\):\s*(\S+)\s*/\s*(\S+)"),
    "dinf":        re.compile(r"Dual infeas \(abs/rel\):\s*(\S+)\s*/\s*(\S+)"),
    "gap":         re.compile(r"Duality gap \(abs/rel\):\s*(\S+)\s*/\s*(\S+)"),
    "phases":      re.compile(r"For LP\s+\S+\s*:\s*Presolve\s+(\S+)\s*\(.*?\)"
                              r"\s*:\s*Solve presolved LP\s+(\S+)\s*\(.*?\)"
                              r"\s*:\s*Postsolve\s+(\S+)"),
    "reductions":  re.compile(r"^Presolve reductions: (.+)$", re.M),
    "maxrss":      re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M),
}


def _f(m, g=1):
    try:
        return float(m.group(g))
    except (AttributeError, ValueError):
        return None


def parse(out: str) -> dict:
    r = {
        "status":     (_RE["status"].search(out).group(1)
                       if _RE["status"].search(out) else None),
        "objective":  _f(_RE["objective"].search(out)),
        "iterations": _f(_RE["iterations"].search(out)),
        "highs_run_time": _f(_RE["highs_time"].search(out)),
        "maxrss_bytes":   _f(_RE["maxrss"].search(out)),
    }
    for k in ("pinf", "dinf", "gap"):
        m = _RE[k].search(out)
        r[k + "_abs"], r[k + "_rel"] = (_f(m, 1), _f(m, 2)) if m else (None, None)
    m = _RE["phases"].search(out)
    if m:
        r["presolve_s"], r["solve_presolved_s"], r["postsolve_s"] = \
            float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = _RE["reductions"].search(out)
    if m:
        r["presolve_reductions"] = m.group(1).strip()
    return r


def run_once(exe: Path, mps: Path, track: str, time_limit: float,
             log_path: Path, env_extra=None) -> dict:
    opts = log_path.with_suffix(".opts")
    opts.write_text(
        "solver=pdlp\n"
        f"presolve={TRACKS[track]['presolve']}\n"
        f"time_limit={time_limit}\n"
        "log_dev_level=1\n"
        "threads=0\n"
    )
    env = dict(os.environ)
    env.update(env_extra or {})
    # /usr/bin/time -l gives peak RSS of the child, which is the number that
    # decides whether an instance fits this machine at all.
    cmd = ["/usr/bin/time", "-l", str(exe), "--options_file", str(opts), str(mps)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=time_limit + 300)
    wall = time.monotonic() - t0
    out = p.stdout + p.stderr
    log_path.write_text(out)
    opts.unlink(missing_ok=True)
    rec = parse(out)
    rec.update(wall=wall, returncode=p.returncode,
               timed_out=bool(rec["status"] and "Time limit" in rec["status"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, action="append")
    ap.add_argument("--only", action="append")
    ap.add_argument("--track", default="A,B")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--highs-src", default=str(Path.home() / "work/cumetal-bench-ext/HiGHS"))
    ap.add_argument("--lp-dir", default=str(HERE / "lps"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--redo", action="store_true", help="ignore existing results")
    a = ap.parse_args()

    src = Path(a.highs_src)
    exes = {"cpu": src / "build-cpu/bin/highs", "gpu": src / "build-gpu/bin/highs"}
    for k, e in exes.items():
        if not e.is_file():
            sys.exit(f"missing {k} build: {e}\nrun scripts/build_highs_cumetal.sh")

    lp_dir, out_dir = Path(a.lp_dir), Path(a.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "runs.jsonl"

    done = set()
    if jsonl.exists() and not a.redo:
        for line in jsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["instance"], d["track"], d["build"], d["rep"]))

    want = instances()
    if a.only:
        want = [i for i in want if i["name"] in a.only]
    else:
        want = [i for i in want if i["phase"] in (a.phase or [1])]
    want = [i for i in want if (lp_dir / f"{i['name']}.mps").is_file()]
    if not want:
        sys.exit("no instances present; run fetch.py first")
    want.sort(key=lambda i: i["nnz"])
    tracks = [t.strip() for t in a.track.split(",")]

    sink = jsonl.open("a")
    for inst in want:
        mps = lp_dir / f"{inst['name']}.mps"
        for track in tracks:
            # One discarded warm-up per (instance, track): the first touch of a
            # multi-hundred-MB MPS pays for page cache that every later run gets
            # free, and that alone can swamp a fast solve.
            todo = [(b, r) for r in range(a.repeats) for b in ("cpu", "gpu")
                    if (inst["name"], track, b, r) not in done]
            if not todo:
                print(f"{inst['name']:<16} {track}  cached", flush=True)
                continue
            warm = out_dir / "logs" / f"{inst['name']}.{track}.warmup.log"
            try:
                run_once(exes["cpu"], mps, track, min(a.time_limit, 120.0), warm)
            except subprocess.TimeoutExpired:
                pass
            for build, rep in todo:
                log = out_dir / "logs" / f"{inst['name']}.{track}.{build}.{rep}.log"
                try:
                    rec = run_once(exes[build], mps, track, a.time_limit, log)
                except subprocess.TimeoutExpired:
                    rec = {"status": "Killed (harness timeout)", "wall": None,
                           "returncode": None, "timed_out": True}
                rec.update(instance=inst["name"], track=track, build=build, rep=rep,
                           rows=inst["rows"], cols=inst["cols"], nnz=inst["nnz"],
                           time_limit=a.time_limit, log=log.name)
                sink.write(json.dumps(rec) + "\n"); sink.flush()
                w = rec.get("wall"); h = rec.get("highs_run_time")
                print(f"{inst['name']:<16} {track} {build} r{rep}  "
                      f"wall={w if w is None else round(w,2):<8} "
                      f"highs={h:<8} {rec.get('status')}", flush=True)
    sink.close()
    print(f"\nresults: {jsonl}")


if __name__ == "__main__":
    sys.exit(main() or 0)
