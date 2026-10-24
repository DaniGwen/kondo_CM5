"""Summarize no-push hip loading in early and settled windows."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent / ('runs/hip_load_20261023_01' + (sys.argv[1] if len(sys.argv) > 1 else ''))
    assert json.loads((folder / 'status.json').read_text())['state'] == 'completed'
    with (folder / 'body.csv').open(newline='') as stream:
        body = list(csv.DictReader(stream))
    modes = {r['episode']: r['mode'] for r in body}
    groups = defaultdict(list)
    with (folder / 'joints.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if 'FemurHead2' not in row['joint']:
                continue
            step = int(row['step'])
            window = 'pre_push_100_124' if 100 <= step < 125 else 'settled_500_625' if 500 <= step <= 625 else None
            if window:
                groups[(modes[row['episode']], window, row['joint'])].append(row)
    lines = ['# Isolated no-push hip load', '',
             'Three clean-reset repetitions per mode, 625 steps (10 seconds), no external pushes. best_then_neutral switches to zero commands after step 125. Percentages pool samples across repetitions.', '',
             '| Mode | Window | Hip | Mean torque usage % | At torque limit % | Tracking error deg | Mean target deg |',
             '|---|---|---|---:|---:|---:|---:|']
    for (mode, window, joint), rows in groups.items():
        mean = lambda values: sum(values) / len(values)
        usage = [abs(float(r['torque_Nm'])) / float(r['max_torque_Nm']) for r in rows]
        error = mean([abs(float(r['server_target_rad']) - float(r['actual_rad'])) for r in rows])
        target = mean([float(r['server_target_rad']) for r in rows])
        lines.append(f'| {mode} | {window} | {joint} | {100*mean(usage):.1f} | {100*mean([float(u >= .95) for u in usage]):.1f} | {math.degrees(error):.3f} | {math.degrees(target):.3f} |')
    lines += ['', f"Falls recorded: {sum(r['fell'] == 'True' for r in body)}."]
    text = '\n'.join(lines)
    (folder / 'findings.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
