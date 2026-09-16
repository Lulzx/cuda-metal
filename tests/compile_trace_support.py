"""Validate diagnostic spans without treating an end record as compile success."""
import math


def parse_spans(stderr, allow_open=False):
    records = []
    opened = {}
    seen = set()
    for line in stderr.splitlines():
        if not line.startswith('CUMETAL_COMPILE '):
            continue
        record = dict(field.split('=', 1) for field in line.split()[1:])
        assert record['event'] in ('begin', 'end'), line
        key = (int(record['pid']), int(record['span']))
        assert key[0] > 0 and key[1] > 0, line
        if record['event'] == 'begin':
            assert key not in seen, ('duplicate span', line)
            assert int(record['input_bytes']) >= 0, line
            opened[key] = record['stage']
            seen.add(key)
        else:
            assert opened.pop(key, None) == record['stage'], ('unpaired end', line)
            duration = float(record['elapsed_ms'])
            assert math.isfinite(duration) and duration >= 0, line
        records.append(record)
    assert allow_open or not opened, ('unclosed spans', opened)
    return records


def begin_stages(records):
    return [record['stage'] for record in records if record['event'] == 'begin']


def without_trace(stderr):
    return '\n'.join(line for line in stderr.splitlines()
                     if not line.startswith('CUMETAL_COMPILE '))
