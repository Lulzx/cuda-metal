#!/usr/bin/env python3
"""Driver-loaded generated MSL must report its actual lowering provenance."""
from pathlib import Path
import subprocess
import sys
from ptx_test_support import run_integer_case
from run_ptx_integer_widths import TEMPLATE

if '--launch' in sys.argv:
    source = TEMPLATE.replace('BODY', 'mov.u64 %rd7, 1;\nmov.u64 %rd8, 2;')
    run_integer_case(Path(sys.argv[1]).resolve(), source, [0], [1, 2], 'provenance probe')
else:
    result = subprocess.run([sys.executable, __file__, sys.argv[1], '--launch'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    launches = [line for line in result.stderr.splitlines()
                if line.startswith('CUMETAL_PROVENANCE event=kernel_launch ')]
    assert launches and all('provenance=generic_ptx_lowering ' in line and
                            'launch_success=true ' in line for line in launches), result.stderr
    print('PROVENANCE_PASS: generated MSL reports generic_ptx_lowering')
