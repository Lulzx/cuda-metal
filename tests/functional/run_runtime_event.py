#!/usr/bin/env python3
"""Compare CUDA event intervals with independent Metal command-buffer durations."""

import os
import re
import subprocess
import sys

result = subprocess.run(
    sys.argv[1:], env={**os.environ, "CUMETAL_TRACE_GPU": "1"},
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print(result.stdout, end="")
if result.returncode:
    sys.exit(result.returncode)

interval = None
kernels = []
checked = 0
for line in result.stdout.splitlines():
    begin = re.fullmatch(r"EVENT_TIMING_BEGIN (\w+)", line)
    end = re.fullmatch(r"EVENT_TIMING_END (\w+) elapsed_ms=(\S+)", line)
    if begin:
        if interval is not None:
            raise SystemExit("FAIL: overlapping event timing intervals")
        interval = begin[1]
        kernels = []
    elif interval and line.startswith("CUMETAL_PROVENANCE event=kernel_launch "):
        duration = re.search(r"duration_ns=(-?\d+)", line)
        if "launch_success=true" not in line or duration is None or int(duration[1]) <= 0:
            raise SystemExit("FAIL: measured kernel lacks successful GPU timing")
        kernels.append(int(duration[1]))
    elif end:
        if interval != end[1] or len(kernels) != 1:
            raise SystemExit("FAIL: event interval must contain one measured GPU kernel")
        elapsed_ms = float(end[2])
        gpu_ms = sum(kernels) / 1e6
        # Float rounding and the GPU timestamp resolution must not hide missing GPU work.
        if elapsed_ms + max(0.001, gpu_ms * 0.001) < gpu_ms:
            raise SystemExit(
                f"FAIL: {interval} CUDA event interval {elapsed_ms:.6f} ms "
                f"does not include Metal GPU execution {gpu_ms:.6f} ms"
            )
        checked += 1
        interval = None
if interval is not None or checked != 3:
    raise SystemExit("FAIL: missing event timing intervals")
print("PASS: CUDA event intervals include independently measured Metal GPU execution")
