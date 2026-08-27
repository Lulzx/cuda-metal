#!/usr/bin/env python3
"""Prove the GPU build actually ran on the Apple GPU for each instance.

A timing table is worthless if the "GPU" build silently fell back to CPU code:
it would report plausible numbers and a ~1.0x speedup, which is exactly what a
broken GPU path looks like. So each instance gets one short traced run
(CUMETAL_TRACE_GPU=1) that must show launches with device=apple_gpu and no
kernel with source=approximate_stub.

Tracing costs far more than the kernels do, so these runs are deliberately not
the timed ones -- they answer "did it run there", not "how fast".

It also reports which kernels ran and how cuPDLP's two sparse products were
routed, since CuMetal's cuSPARSE picks CPU or Metal per call from the row
distribution and that choice is the main thing driving these timings.
"""
import argparse, json, os, re, subprocess, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch import instances  # noqa: E402

KERNEL = re.compile(r'kernel="([^"]+)"')
DEVICE = re.compile(r"device=(\S+)")
QUALITY = re.compile(r"semantic_quality=(\S+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, action="append")
    ap.add_argument("--only", action="append")
    ap.add_argument("--time-limit", type=float, default=20.0,
                    help="short on purpose; this checks provenance, not speed")
    ap.add_argument("--highs-src", default=str(Path.home() / "work/cumetal-bench-ext/HiGHS"))
    ap.add_argument("--lp-dir", default=str(HERE / "lps"))
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    exe = Path(a.highs_src) / "build-gpu/bin/highs"
    if not exe.is_file():
        sys.exit(f"missing gpu build: {exe}")
    lp_dir, out_dir = Path(a.lp_dir), Path(a.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    want = instances()
    if a.only:
        want = [i for i in want if i["name"] in a.only]
    else:
        want = [i for i in want if i["phase"] in (a.phase or [1])]
    want = [i for i in want if (lp_dir / f"{i['name']}.mps").is_file()]
    want.sort(key=lambda i: i["nnz"])

    env = dict(os.environ, CUMETAL_TRACE_GPU="1", CUMETAL_DEBUG_SPARSE="1")
    results, fails = [], 0
    for inst in want:
        opts = out_dir / "logs" / f"{inst['name']}.verify.opts"
        opts.write_text(f"solver=pdlp\npresolve=off\ntime_limit={a.time_limit}\n")
        log = out_dir / "logs" / f"{inst['name']}.verify.log"
        t0 = time.monotonic()
        p = subprocess.run([str(exe), "--options_file", str(opts),
                            str(lp_dir / f"{inst['name']}.mps")],
                           capture_output=True, text=True, env=env,
                           timeout=a.time_limit + 600)
        out = p.stdout + p.stderr
        log.write_text(out)
        opts.unlink(missing_ok=True)

        devices = Counter(DEVICE.findall(out))
        kernels = Counter(KERNEL.findall(out))
        quality = Counter(QUALITY.findall(out))
        gpu = devices.get("apple_gpu", 0)
        stub = "source=approximate_stub" in out
        ok = gpu > 0 and not stub
        fails += 0 if ok else 1
        rec = dict(instance=inst["name"], nnz=inst["nnz"], apple_gpu_launches=gpu,
                   other_devices={k: v for k, v in devices.items() if k != "apple_gpu"},
                   approximate_stub=stub, semantic_quality=dict(quality),
                   top_kernels=dict(kernels.most_common(6)),
                   elapsed=round(time.monotonic() - t0, 1), ok=ok)
        results.append(rec)
        spmv = sum(v for k, v in kernels.items() if "spmv" in k)
        print(f"{inst['name']:<16} apple_gpu={gpu:<8} spmv={spmv:<7} "
              f"{'OK' if ok else 'FAIL'}"
              f"{'  (approximate_stub!)' if stub else ''}", flush=True)

    (out_dir / "provenance.json").write_text(json.dumps(results, indent=2))
    print(f"\n{len(results)-fails}/{len(results)} instances confirmed on the Apple GPU")
    print(f"written: {out_dir / 'provenance.json'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
