#!/usr/bin/env python3
"""Host-only trace pairing, opt-in, interruption, and PTX-stage regression tests."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compile_trace_support import begin_stages, parse_spans, without_trace


def run(command, setting):
    env = os.environ.copy()
    env.pop('CUMETAL_TRACE_COMPILE', None)
    if setting is not None:
        env['CUMETAL_TRACE_COMPILE'] = setting
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)


def main():
    probe, compiler = sys.argv[1:]
    for setting in (None, '', '0', 'true'):
        result = run([probe, 'nested'], setting)
        assert result.returncode == 0 and result.stderr == '', result
    nested = run([probe, 'nested'], '1')
    assert nested.returncode == 0, nested.stderr
    records = parse_spans(nested.stderr)
    assert [(r['event'], r['stage']) for r in records] == [
        ('begin', 'outer'), ('begin', 'inner'), ('end', 'inner'), ('end', 'outer')]
    assert [r['input_bytes'] for r in records[:2]] == ['64', '8']
    concurrent = run([probe, 'concurrent'], '1')
    assert concurrent.returncode == 0, concurrent.stderr
    records = parse_spans(concurrent.stderr)
    assert begin_stages(records) == ['parallel'] * 8, concurrent.stderr
    assert all(r['event'] == 'begin' for r in records[:8]), concurrent.stderr
    interrupted = run([probe, 'interrupted'], '1')
    assert interrupted.returncode == 42, interrupted.stderr
    records = parse_spans(interrupted.stderr, allow_open=True)
    assert len(records) == 1 and records[0]['event'] == 'begin', interrupted.stderr

    with tempfile.TemporaryDirectory(prefix='cumetal-trace-host-') as work:
        source, output = Path(work) / 'probe.ptx', Path(work) / 'probe.metal'
        source.write_text('.version 7.0\n.target sm_80\n.address_size 64\n'
                          '.visible .entry trace_probe() { ret; }\n')
        command = [compiler, str(source), '--backend=cumetal-ir', '--ptx-strict',
                   '--entry', 'trace_probe', '--emit=msl', '--overwrite', '-o', str(output)]
        baseline = run(command, None)
        assert baseline.returncode == 0, baseline.stderr
        assert parse_spans(baseline.stderr) == []
        expected_source = output.read_bytes()
        traced = run(command, '1')
        assert traced.returncode == 0, traced.stderr
        assert output.read_bytes() == expected_source
        assert without_trace(traced.stderr) == without_trace(baseline.stderr)
        records = parse_spans(traced.stderr)
        # The PTX importer traces its own phases as well, so this checks the
        # backbone stages appear in this order rather than pinning the full
        # list -- which would turn any new phase into a test failure.
        stages = begin_stages(records)
        backbone = [stage for stage in stages
                    if stage in ('ptx_to_msl_total', 'ptx_import',
                                 'metal_legalization', 'msl_generation')]
        assert backbone == [
            'ptx_to_msl_total', 'ptx_import', 'metal_legalization', 'msl_generation'], stages
        assert records[0]['input_bytes'] == str(source.stat().st_size)
        assert records[-1]['stage'] == 'ptx_to_msl_total'

        source.write_text(source.read_text().replace('ret;', 'unsupported.trace.op; ret;'))
        failed = run(command, None)
        traced_failure = run(command, '1')
        assert failed.returncode != 0 and traced_failure.returncode == failed.returncode
        assert without_trace(traced_failure.stderr) == without_trace(failed.stderr)
        failure_stages = begin_stages(parse_spans(traced_failure.stderr))
        assert failure_stages[:2] == ['ptx_to_msl_total', 'ptx_import'], traced_failure.stderr
        assert 'msl_generation' not in failure_stages, traced_failure.stderr
    print('PASS: compile trace opt-in, nesting, concurrency, interruption, PTX success/failure')


if __name__ == '__main__':
    main()
