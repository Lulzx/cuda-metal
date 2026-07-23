#!/usr/bin/env python3
"""Build and run every CUDA-project fixture with failure classification."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


CLASSIFICATIONS = {
    "pass",
    "prerequisite_skip",
    "compile_error",
    "link_error",
    "unsupported_kernel",
    "numerical_failure",
    "crash",
    "timeout",
    "runtime_error",
}


def classify(returncode: int, output: str, timed_out: bool = False) -> str:
    text = output.lower()
    if timed_out:
        return "timeout"
    if "registered kernel missing metallib" in text or "missing metallib" in text:
        return "unsupported_kernel"
    if (
        "undefined symbols for architecture" in text
        or "linker command failed" in text
        or "ld: " in text and "error" in text
    ):
        return "link_error"
    if (
        "error generated when compiling for" in text
        or "clang: error:" in text
        or "fatal error:" in text
        or "compilation failed" in text
    ):
        return "compile_error"
    if (
        "numerically wrong" in text
        or "mismatch" in text
        or "max error" in text
        or "fail:" in text
    ):
        return "numerical_failure"
    if returncode < 0 or returncode in (134, 137, 139) or "abort trap" in text:
        return "crash"
    if returncode == 77 or "skip:" in text:
        return "prerequisite_skip"
    if returncode == 0:
        return "pass"
    return "runtime_error"


def self_test() -> int:
    cases = [
        (0, "PASS", False, "pass"),
        (77, "SKIP: clang++ not found", False, "prerequisite_skip"),
        (77, "registered kernel missing metallib", False, "unsupported_kernel"),
        (1, "clang: error: unsupported option", False, "compile_error"),
        (1, "Undefined symbols for architecture arm64", False, "link_error"),
        (77, "FAIL: mismatch at 7", False, "numerical_failure"),
        (139, "Segmentation fault: 11", False, "crash"),
        (1, "", True, "timeout"),
        (2, "unexpected exit", False, "runtime_error"),
    ]
    for returncode, output, timed_out, expected in cases:
        actual = classify(returncode, output, timed_out)
        if actual != expected:
            print(
                f"classification mismatch: expected {expected}, got {actual}",
                file=sys.stderr,
            )
            return 1
    print("CUDA-project sweep classifier tests passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        choices=sorted(CLASSIFICATIONS - {"pass"}),
        help="return nonzero if this classification is observed",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, fixtures_root: Path) -> list[dict[str, str]]:
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    projects = document.get("projects") if isinstance(document, dict) else None
    if not isinstance(projects, list) or not projects:
        raise ValueError("manifest must contain a non-empty projects array")

    required = {"name", "subdir", "source", "binary", "harness"}
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    sources: set[Path] = set()
    for index, raw in enumerate(projects):
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"manifest project {index} is missing required fields")
        project = {key: str(raw[key]) for key in required}
        if project["name"] in names:
            raise ValueError(f"duplicate project name: {project['name']}")
        if project["harness"] not in ("standard", "strict"):
            raise ValueError(f"invalid harness for {project['name']}")
        source = fixtures_root / project["subdir"] / project["source"]
        if not source.is_file():
            raise ValueError(f"missing source for {project['name']}: {source}")
        names.add(project["name"])
        sources.add(source.resolve())
        normalized.append(project)

    discovered = {
        source.resolve()
        for source in fixtures_root.glob("*/*.cu")
        if "build" not in source.parts
    }
    unlisted = sorted(discovered - sources)
    if unlisted:
        relative = ", ".join(str(path.relative_to(fixtures_root)) for path in unlisted)
        raise ValueError(f"unmanifested CUDA fixtures: {relative}")
    return normalized


def run_command(
    command: list[str], cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int, str, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
        return process.returncode, output, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return 1, output, True


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.timeout <= 0:
        print("timeout must be positive", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    fixtures_root = root / "tests/cuda_projects"
    manifest = args.manifest or fixtures_root / "coverage_manifest.json"
    build_dir = (args.build_dir or root / "build/tests/cuda_projects").resolve()
    output_dir = (args.output_dir or root / "build/cuda-project-sweep").resolve()
    try:
        projects = load_manifest(manifest, fixtures_root)
    except ValueError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2

    selected = set(args.project)
    known = {project["name"] for project in projects}
    unknown = selected - known
    if unknown:
        print(f"unknown projects: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if selected:
        projects = [project for project in projects if project["name"] in selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env.setdefault("CUMETAL_BUILD_DIR", str(root / "build"))
    env["CUMETAL_CUDA_PROJECT_STRICT_CLASSIFICATION"] = "1"
    for project in projects:
        harness_name = (
            "run_strict_standalone_cu.sh"
            if project["harness"] == "strict"
            else "run_standalone_cu.sh"
        )
        command = [
            "bash",
            str(fixtures_root / harness_name),
            str(root),
            str(build_dir),
            project["subdir"],
            project["source"],
            project["binary"],
        ]
        started = time.monotonic()
        returncode, output, timed_out = run_command(
            command, root, env, args.timeout
        )
        elapsed = time.monotonic() - started
        classification = classify(returncode, output, timed_out)
        log_path = output_dir / f"{project['name']}.log"
        log_path.write_text(output, encoding="utf-8")
        row = {
            **project,
            "classification": classification,
            "return_code": "timeout" if timed_out else returncode,
            "elapsed_seconds": round(elapsed, 3),
            "log": str(log_path),
        }
        rows.append(row)
        print(
            f"{project['name']}\t{classification}\trc={row['return_code']}"
            f"\t{elapsed:.2f}s\t{log_path}",
            flush=True,
        )

    summary_path = output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write(
            "project\tsource\tclassification\treturn_code\telapsed_seconds\tlog\n"
        )
        for row in rows:
            summary.write(
                f"{row['name']}\t{row['subdir']}/{row['source']}\t"
                f"{row['classification']}\t{row['return_code']}\t"
                f"{row['elapsed_seconds']:.3f}\t{row['log']}\n"
            )
    counts = Counter(row["classification"] for row in rows)
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps({"counts": dict(sorted(counts.items())), "projects": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    print("Classifications: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    return 1 if any(counts[status] for status in args.fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
