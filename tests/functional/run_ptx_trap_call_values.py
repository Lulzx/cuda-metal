#!/usr/bin/env python3
"""Trap-capable helper returns preserve local writes and distinct call sites."""
from pathlib import Path
import random
import sys
from ptx_test_support import run_integer_case

rng = random.Random(47238)
values = [0, 1, 255, 1 << 32, 1 << 59] + [rng.getrandbits(60) for _ in range(256)]
expected = [word for value in values for word in (value + 15, value + 3)]
source = (Path(__file__).parent / 'reference/ptx_trap_call_values.ptx').read_text()
run_integer_case(Path(sys.argv[1]).resolve(), source, values, expected,
                 'trap-capable helper values and side effects', entry='tail_probe')
