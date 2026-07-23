#!/usr/bin/env python3
"""Sweep llama.cpp GPU offload counts and classify the first failing boundary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def classify(returncode: int, output: str, timed_out: bool = False) -> str:
    text = output.lower()
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "pass"
    if returncode == 77:
        return "skip"
    if "incoherent output" in text or "numerically wrong output" in text:
        return "incoherent_output"
    if "cublas_status" in text or "cublas" in text and "failed" in text:
        return "cublas_error"
    if "registered kernel missing metallib" in text or "missing metallib" in text:
        return "unsupported_kernel"
    if returncode < 0 or returncode in (134, 137, 139) or "abort trap" in text:
        return "crash"
    return "runtime_error"


def self_test() -> int:
    cases = [
        (0, "PASS", False, "pass"),
        (77, "SKIP", False, "skip"),
        (1, "FAIL: incoherent output — numerically wrong output", False, "incoherent_output"),
        (1, "CUBLAS_STATUS_INVALID_VALUE", False, "cublas_error"),
        (1, "registered kernel missing metallib", False, "unsupported_kernel"),
        (134, "Abort trap: 6", False, "crash"),
        (1, "", True, "timeout"),
        (2, "unknown failure", False, "runtime_error"),
    ]
    for returncode, output, timed_out, expected in cases:
        actual = classify(returncode, output, timed_out)
        if actual != expected:
            print(f"classification mismatch: expected {expected}, got {actual}", file=sys.stderr)
            return 1
    print("llama NGL sweep classifier tests passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-ngl", type=int, default=1)
    parser.add_argument("--max-ngl", type=int, default=99)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per run")
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.min_ngl < 0 or args.max_ngl < args.min_ngl or args.timeout <= 0:
        print("invalid NGL range or timeout", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    harness = args.harness or root / "tests/conformance/run_llama_cpp_cumetal.sh"
    output_dir = args.output_dir or root / "build/llama-ngl-sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.tsv"

    rows: list[tuple[int, str, int | str, float, Path]] = []
    boundary: tuple[int, str] | None = None
    for ngl in range(args.min_ngl, args.max_ngl + 1):
        env = os.environ.copy()
        env["CUMETAL_LLAMA_NGL"] = str(ngl)
        # Keep enough decode tokens for the default factual probe to reach
        # "Paris". A one-token run produces "The" even on the known-good NGL=1
        # path and would therefore manufacture a false coherence boundary.
        env.setdefault("CUMETAL_LLAMA_NTOK", "16")
        started = time.monotonic()
        timed_out = False
        try:
            result = subprocess.run(
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
            returncode: int | str = result.returncode
            output = result.stdout
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = "timeout"
            captured = exc.stdout or ""
            output = captured.decode(errors="replace") if isinstance(captured, bytes) else captured

        elapsed = time.monotonic() - started
        status = classify(returncode if isinstance(returncode, int) else 1, output, timed_out)
        log_path = output_dir / f"ngl-{ngl:02d}.log"
        log_path.write_text(output, encoding="utf-8")
        rows.append((ngl, status, returncode, elapsed, log_path))
        print(f"NGL={ngl:02d}\t{status}\trc={returncode}\t{elapsed:.2f}s\t{log_path}", flush=True)

        if status not in ("pass", "skip") and boundary is None:
            boundary = (ngl, status)
            if not args.continue_after_failure:
                break

    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("ngl\tclassification\treturn_code\telapsed_seconds\tlog\n")
        for ngl, status, returncode, elapsed, log_path in rows:
            summary.write(f"{ngl}\t{status}\t{returncode}\t{elapsed:.3f}\t{log_path}\n")

    if boundary:
        print(f"First failing boundary: NGL={boundary[0]} ({boundary[1]})")
        print(f"Summary: {summary_path}")
        return 1
    print(f"No failure found from NGL={args.min_ngl} through NGL={args.max_ngl}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
