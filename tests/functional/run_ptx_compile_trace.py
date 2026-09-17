#!/usr/bin/env python3
"""Host-only PTX trace pairing and compilation-equivalence regression."""
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote


PREFIX = 'CUMETAL_COMPILE '
ENTRY = 'trace$kernel'
HELPER = 'trace$helper'
PTX = '''.version 7.1
.target sm_80
.address_size 64
.func (.param .b32 result) trace$helper(.param .b32 value) {
 .reg .b32 %r0;
 ld.param.b32 %r0, [value];
 add.u32 %r0, %r0, 7;
 st.param.b32 [result], %r0;
 ret;
}
.visible .entry trace$kernel(.param .u64 .ptr .global output, .param .u32 value) {
 .reg .b64 %rd0;
 .reg .b32 %r0, %r1;
 .param .b32 arg;
 .param .b32 result;
 ld.param.u64 %rd0, [output];
 ld.param.u32 %r0, [value];
 st.param.b32 [arg], %r0;
 call.uni (result), trace$helper, (arg);
 ld.param.b32 %r1, [result];
 st.global.b32 [%rd0], %r1;
 ret;
}
'''


def records(stderr):
    spans, opened, seen = [], {}, set()
    for line in stderr.splitlines():
        if not line.startswith(PREFIX):
            continue
        fields = [field.split('=', 1) for field in line.split()[1:]]
        record = dict(fields)
        assert len(record) == len(fields), ('duplicate trace field', line)
        key = (int(record['pid']), int(record['span']))
        assert key[0] > 0 and key[1] > 0, line
        context = (record['stage'], record.get('function'))
        if context[1] is not None:
            assert context[1] in (quote(ENTRY, safe=''), quote(HELPER, safe='')), line
        if record['event'] == 'begin':
            assert key not in seen and int(record['input_bytes']) >= 0, line
            opened[key] = context
            seen.add(key)
        else:
            assert record['event'] == 'end' and opened.pop(key, None) == context, line
            duration = float(record['elapsed_ms'])
            assert math.isfinite(duration) and duration >= 0, line
        spans.append(record)
    assert not opened, ('unclosed trace spans', opened)
    return spans


def without_trace(stderr):
    return ''.join(line for line in stderr.splitlines(keepends=True)
                   if not line.startswith(PREFIX))


def compile_ptx(command, setting):
    env = os.environ.copy()
    env.pop('CUMETAL_TRACE_COMPILE', None)
    if setting is not None:
        env['CUMETAL_TRACE_COMPILE'] = setting
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)


def main():
    if not __debug__:
        raise RuntimeError('This test requires Python assertions')
    build = Path(sys.argv[1] if len(sys.argv) > 1 else 'build').resolve()
    with tempfile.TemporaryDirectory(prefix='cumetal-ptx-trace-') as work:
        source, output = Path(work) / 'probe.ptx', Path(work) / 'probe.metal'
        abi = Path(str(output) + '.cumetal-abi')
        command = [str(build / 'cumetalc'), str(source), '--backend=cumetal-ir',
                   '--ptx-strict', '--entry', ENTRY, '--emit=msl', '--overwrite',
                   '-o', str(output)]
        source.write_text(PTX)
        baseline = compile_ptx(command, None)
        assert baseline.returncode == 0, baseline.stderr
        assert records(baseline.stderr) == []
        expected_output, expected_abi = output.read_bytes(), abi.read_bytes()
        assert expected_output and expected_abi.startswith(b'CUMETAL_ABI_V2\n')
        for setting in ('0', 'false', '1'):
            compiled = compile_ptx(command, setting)
            assert compiled.returncode == baseline.returncode, compiled.stderr
            assert compiled.stdout == baseline.stdout, compiled.stdout
            assert without_trace(compiled.stderr) == baseline.stderr, compiled.stderr
            assert output.read_bytes() == expected_output, 'tracing changed MSL'
            assert abi.read_bytes() == expected_abi, 'tracing changed ABI'
            spans = records(compiled.stderr)
            if setting != '1':
                assert not spans, (setting, spans)
                continue
            begins = [record for record in spans if record['event'] == 'begin']
            stages = {record['stage'] for record in begins}
            assert {'ptx_import', 'ptx_parse', 'ptx_scan_threadgroup_globals',
                    'ptx_scan_local_depots', 'ptx_scan_implicit_definitions',
                    'ptx_scan_initialized_arrays', 'ptx_scan_module_constants',
                    'ptx_scan_module_globals', 'ptx_verify'} <= stages, stages
            assert any(record['stage'] == 'ptx_parse' and
                       int(record['input_bytes']) == len(PTX.encode()) for record in begins)
            for name in (ENTRY, HELPER):
                stages = {record['stage'] for record in begins
                          if record.get('function') == quote(name, safe='')}
                assert {'ptx_normalize_function', 'ptx_printf_scaffold', 'ptx_printf_lower',
                        'ptx_import_function', 'ptx_cfg_normalization', 'ptx_ssa',
                        'ptx_resolve_types', 'ptx_materialization'} <= stages, (name, stages)

        output.unlink()
        abi.unlink()
        source.write_text(PTX.replace('add.u32 %r0, %r0, 7;', 'unsupported.trace.op;'))
        failed = compile_ptx(command, None)
        assert failed.returncode != 0 and 'unsupported opcode' in failed.stderr, failed.stderr
        assert records(failed.stderr) == []
        assert set(Path(work).iterdir()) == {source}, 'failed compilation left artifacts'
        for setting in ('0', 'false', '1'):
            compiled = compile_ptx(command, setting)
            assert compiled.returncode == failed.returncode, compiled.stderr
            assert compiled.stdout == failed.stdout, compiled.stdout
            assert without_trace(compiled.stderr) == failed.stderr, compiled.stderr
            assert set(Path(work).iterdir()) == {source}, 'failed compilation left artifacts'
            spans = records(compiled.stderr)
            if setting == '1':
                assert spans and spans[0]['event'] == 'begin', spans
                assert spans[-1]['event'] == 'end' and spans[-1]['stage'] == 'ptx_import', spans
                assert any(record['stage'] == 'ptx_call_graph' and
                           record.get('function') == quote(ENTRY, safe='') for record in spans)
            else:
                assert not spans, (setting, spans)
    print('PASS: PTX compile trace pairing, escaped context, output and diagnostic equivalence')


if __name__ == '__main__':
    main()
