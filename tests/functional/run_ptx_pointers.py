#!/usr/bin/env python3
"""Local helper and pointer arithmetic regressions using the shared runner."""
from pathlib import Path
import random
import sys
from ptx_test_support import expect_compile_failure, run_integer_case

REFERENCE = Path(__file__).parent / 'reference'


def main(build):
    rng = random.Random(19058)
    values = [0, 1, (1 << 64)-1, 0x0123456789abcdef, 1 << 63]
    values += [rng.getrandbits(64) for _ in range(256)]
    helper = (REFERENCE / 'ptx_local_helper.ptx').read_text()
    run_integer_case(build, helper, values, values, 'local helper',
                     entry='local_helper_probe', output_words=1)
    dead_cast = helper.replace('cvta.local.u64 %rd9, %rd7;',
                              'cvta.local.u64 %rd9, %rd7;\ncvt.u32.u64 %r5, %rd7;')
    run_integer_case(build, dead_cast, values, values, 'unused pointer truncation',
                     entry='local_helper_probe', output_words=1)
    for argument in ('7', '%rd1'):
        invalid = helper.replace('st.param.b64 [arg], %rd9;',
                                 f'st.param.b64 [arg], {argument};')
        expect_compile_failure(build, invalid, 'local_helper_probe', '')

    source = (REFERENCE / 'ptx_pointer_subtract.ptx').read_text()
    expected = [word for value in values for word in
                (int.from_bytes(value.to_bytes(8, 'little'), 'big'), value)]
    run_integer_case(build, source, values, expected, 'pointer subtraction', entry='tail_probe')
    for invalid in ('sub.u64 %rd12, %rd9, %rd8;',
                    'sub.u64 %rd12, %rd10, %rd9;',
                    'sub.u32 %rd12, %rd9, %rd10;'):
        expect_compile_failure(build,
            source.replace('sub.u64 %rd12, %rd9, %rd10;', invalid),
            'tail_probe', 'pointer subtraction')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
