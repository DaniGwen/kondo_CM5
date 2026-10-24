"""Summarize outcomes, hip loading, and foot displacement across friction settings."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    output = Path(__file__).resolve().parent / 'runs/friction_comparison_20261023_01'
    assert json.loads((output / 'status.json').read_text())['state'] == 'completed'
    lines = ['# Friction comparison', '',
             '360 deterministic push trials, five repetitions per force/direction/policy/friction. Clean reset, 160 ms pulses at step 125, 2 degree / 10 consecutive step recovery threshold. Explicit default/default contact coefficients in separate world copies. No training or production-world edits.', '',
             '| Friction | Policy | Force N | Recovered / 20 | Falls / 20 | Mean peak tilt deg |',
             '|---:|---|---:|---:|---:|---:|']
    telemetry = []
    for label, friction in [('1', 1), ('0p5', .5), ('0p25', .25)]:
        folder = output / ('mu_' + label)
        with (folder / 'trials.csv').open(newline='') as stream:
            trials = list(csv.DictReader(stream))
        mapping = {r['episode']: r for r in trials}
        for mode in ('neutral', 'best_188227'):
            for force in ('4', '6', '8'):
                rows = [r for r in trials if r['checkpoint'] == mode and r['force_N'] == force]
                peak = sum(float(r['peak_tilt_deg']) for r in rows) / len(rows)
                lines.append(f"| {friction} | {mode} | {force} | {sum(r['outcome'] != 'failed' for r in rows)} | {sum(r['fell'] == 'True' for r in rows)} | {peak:.2f} |")
        load = defaultdict(list)
        with (folder / 'measured_joints.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                if 'FemurHead2' in row['joint']:
                    step = int(row['step'])
                    window = 'pre' if 100 <= step < 125 else 'post' if step >= 125 else None
                    if window:
                        load[(row['episode'], window)].append(abs(float(row['torque_Nm'])) / float(row['max_torque_Nm']))
        feet = defaultdict(list)
        with (folder / 'contacts.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                feet[(row['episode'], row['side'])].append(row)
        for episode, meta in mapping.items():
            displacements = []
            for side in ('Left', 'Right'):
                rows = feet[(episode, side)]
                before = [r for r in rows if int(r['step']) == 124]
                after = [r for r in rows if int(r['step']) >= 125]
                if before and after:
                    origin = before[0]
                    displacements.append(max(math.hypot(float(r['foot_x']) - float(origin['foot_x']), float(r['foot_y']) - float(origin['foot_y'])) for r in after))
            mean = lambda seq: sum(seq) / len(seq) if seq else 0
            telemetry.append(dict(friction=friction, **meta,
                pre_hip_torque_pct=100*mean(load[(episode, 'pre')]),
                post_hip_torque_pct=100*mean(load[(episode, 'post')]),
                max_foot_displacement_mm=1000*max(displacements, default=0)))
    with (output / 'trial_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(telemetry[0]))
        writer.writeheader()
        writer.writerows(telemetry)
    lines += ['', '## Best-policy loading and displacement', '',
              'Hip torque is mean absolute feedback divided by available torque, averaged across both FemurHead2 joints. Foot displacement is maximum horizontal foot movement from step 124, averaged across trials. It includes toppling and foot lifting, so it is not a direct sliding measurement.', '',
              '| Friction | Force N | Pre-push hip torque % | Post-push hip torque % | Foot displacement mm |',
              '|---:|---:|---:|---:|---:|']
    for friction in (1, .5, .25):
        for force in ('4', '6', '8'):
            rows = [r for r in telemetry if r['friction'] == friction and r['force_N'] == force and r['checkpoint'] == 'best_188227']
            values = [sum(r[key] for r in rows) / len(rows) for key in ('pre_hip_torque_pct', 'post_hip_torque_pct', 'max_foot_displacement_mm')]
            lines.append(f'| {friction} | {force} | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} |')
    text = '\n'.join(lines) + '\n'
    (output / 'findings.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
