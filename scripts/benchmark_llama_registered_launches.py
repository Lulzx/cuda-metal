#!/usr/bin/env python3
"""Compare fenced asynchronous and host-synchronized llama.cpp launches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--ngl", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.ngl < 0 or args.timeout <= 0:
        parser.error("invalid repetitions, warmups, NGL, or timeout")

    root = Path(__file__).resolve().parents[1]
    harness = root / "tests/conformance/run_llama_cpp_cumetal.sh"
    output_dir = args.output_dir or root / "build/llama-registered-launch-bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = (
        ("fenced_async", False),
        ("host_synchronized", True),
    )
    results: dict[str, list[float]] = {name: [] for name, _ in modes}

    for iteration in range(args.warmups + args.repetitions):
        for name, force_sync in modes:
            env = os.environ.copy()
            env["CUMETAL_LLAMA_NGL"] = str(args.ngl)
            env.setdefault("CUMETAL_LLAMA_NTOK", "16")
            if force_sync:
                env["CUMETAL_SYNC_REGISTERED_LAUNCH"] = "1"
            else:
                env.pop("CUMETAL_SYNC_REGISTERED_LAUNCH", None)

            started = time.monotonic()
            completed = subprocess.run(
                ["bash", str(harness)],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
            elapsed = time.monotonic() - started
            phase = "warmup" if iteration < args.warmups else f"run-{iteration - args.warmups + 1}"
            log_path = output_dir / f"{name}-{phase}.log"
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or "PASS: llama.cpp produced correct output" not in completed.stdout:
                print(f"FAIL: {name} {phase} failed; see {log_path}")
                return 1
            print(f"{name}\t{phase}\t{elapsed:.3f}s", flush=True)
            if iteration >= args.warmups:
                results[name].append(elapsed)

    summary = {
        name: {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }
        for name, samples in results.items()
    }
    async_median = summary["fenced_async"]["median_seconds"]
    sync_median = summary["host_synchronized"]["median_seconds"]
    summary["fenced_async"]["ratio_vs_host_synchronized"] = async_median / sync_median
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"median fenced_async={async_median:.3f}s "
        f"host_synchronized={sync_median:.3f}s "
        f"ratio={async_median / sync_median:.2f}x"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
