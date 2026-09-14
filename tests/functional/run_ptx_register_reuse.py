#!/usr/bin/env python3
"""An offset and later pointer in the same PTX register have distinct SSA types."""
from pathlib import Path
import struct
import sys
from ptx_test_support import run_integer_case

build = Path(sys.argv[1]).resolve()
source = (Path(__file__).parent / 'reference/ptx_register_reuse.ptx').read_text()
values = [float(i - 32) / 4 for i in range(65)]
bits = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]
abi = ['CUMETAL_ABI_V2', 'kernel clamp_relu', 'shared 0',
       'arg buffer 8', 'arg buffer 8', 'arg bytes 4']
for backend in ('legacy', 'cumetal-ir'):
    run_integer_case(build, source, list(map(bits, values)),
                     [bits(max(value, 0.0)) for value in values],
                     backend + ' reused scalar/pointer registers', entry='clamp_relu',
                     word_bits=32, output_words=1, backend=backend, abi_lines=abi)
