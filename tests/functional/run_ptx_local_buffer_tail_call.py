#!/usr/bin/env python3
"""Local-frame tail recursion: runtime input, zero fallback, and proof boundaries."""
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from run_ptx_scalar_tail_call import run_case


def main():
    build = Path(sys.argv[1]).resolve()
    ptx = (Path(__file__).parent / 'reference/ptx_local_buffer_tail_call.ptx').read_text()
    rng = random.Random(1907)
    pairs = [(0, 0), (1, 0), (0, 1), ((1 << 64)-1, (1 << 64)-1)]
    pairs += [(1 << bit, 0) for bit in range(64)]
    pairs += [(0, 1 << bit) for bit in range(64)]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(256)]
    # Independent seed_from_u64(0) oracle, rather than copying PTX constants.
    mask = (1 << 64) - 1
    def splitmix(value):
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return value ^ (value >> 31)
    fallback = tuple(splitmix((i * 0x9e3779b97f4a7c15) & mask) for i in (1, 2))
    expected = [word for pair in pairs for word in (fallback if pair == (0, 0) else pair)]
    run_case(build, ptx, [word for pair in pairs for word in pair], expected,
             'LLVM 19 from_seed: full input preservation and zero-seed fallback')

    # Exact source fragments ensure these cases change the intended proof edge.
    mutations = [
        ('[%rd2+15]', '[%rd2+14]'),  # incomplete input consumption
        ('[%rd2+15]', '[%rd2+16]'),  # out-of-bounds input
        ('st.local.v2.b64', 'st.local.b64'),  # incomplete replacement
        ('[%rd49]', '[%rd49+8]'),  # replacement must cover the same full frame
        ('add.u64 \t%rd48, %SP, 0;', 'add.u64 \t%rd48, %SP, 8;'),
        ('or.b64 \t%rd51, %rd23, %rd12;', 'or.b64 \t%rd51, %SP, %rd12;'),  # escaping address bits
        ('st.local.v2.b64', '@%p1 st.local.v2.b64'),
        ('ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];',
         'ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];\nadd.u64 %rd50, %rd50, 1;'),
        ('ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];',
         'ld.param.v2.b64 \t{%rd50, %rd51}, [retval0];\nld.local.b8 %rd3, [%rd2];'),
    ]
    with tempfile.TemporaryDirectory(prefix='cumetal-local-tail-negative-') as work:
        source, output = Path(work) / 'test.ptx', Path(work) / 'test.metal'
        for old, new in mutations:
            assert old in ptx, old
            source.write_text(ptx.replace(old, new, 1))
            result = subprocess.run([str(build / 'cumetalc'), str(source), '--backend=cumetal-ir',
                                     '--ptx-strict', '--entry', 'tail_probe', '--emit=msl',
                                     '--overwrite', '-o', str(output)], capture_output=True, text=True)
            if result.returncode == 0 or 'recursive PTX device-call cycle' not in result.stderr:
                raise RuntimeError(f'unsafe tail rewrite accepted or wrong failure: {old}\n{result.stderr}')
    print(f'NEGATIVE_PASS {len(mutations)} local-frame proof boundaries')


if __name__ == '__main__':
    main()
