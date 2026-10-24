"""Append joint-load and direction summaries to the completed matched evaluation."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent / 'runs/sac_box_torque125_evaluation_20261023_01'
    assert json.loads((folder / 'status.json').read_text())['state'] == 'completed'
    with (folder / 'trials.csv').open(newline='') as stream:
        trials = list(csv.DictReader(stream))
    mapping = {r['episode']: r for r in trials}
    groups = defaultdict(list)
    with (folder / 'measured_joints.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['joint'].endswith(('FemurHead2', 'Crus')):
                groups[(row['episode'], row['joint'])].append(row)
    rows = []
    mean = lambda values: sum(values) / len(values)
    for (episode, joint), samples in groups.items():
        before = [r for r in samples if 100 <= int(r['step']) < 125]
        after = [r for r in samples if int(r['step']) >= 125]
        if not before or not after:
            continue
        origin = mean([float(r['actual_rad']) for r in before])
        target = mean([float(r['server_target_rad']) for r in before])
        rows.append(dict(**mapping[episode], joint=joint,
            pre_torque_pct=100*mean([abs(float(r['torque_Nm'])) / float(r['max_torque_Nm']) for r in before]),
            post_torque_pct=100*mean([abs(float(r['torque_Nm'])) / float(r['max_torque_Nm']) for r in after]),
            target_excursion_deg=math.degrees(max(abs(float(r['server_target_rad']) - target) for r in after)),
            actual_excursion_deg=math.degrees(max(abs(float(r['actual_rad']) - origin) for r in after)),
            tracking_error_deg=math.degrees(mean([abs(float(r['actual_rad']) - float(r['server_target_rad'])) for r in after]))))
    with (folder / 'joint_trial_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['', '## Joint response at 6 N', '',
             'Mean across twenty trials per policy. Torque usage is absolute feedback divided by the increased available limit. Excursion is maximum change from the pre-push mean; actual excursion can include passive motion.', '',
             '| Policy | Joint | Pre-push torque % | Post-push torque % | Target excursion deg | Actual excursion deg | Tracking error deg |',
             '|---|---|---:|---:|---:|---:|---:|']
    for mode in ('neutral', 'previous_best', 'new_best'):
        for joint in ('RightFemurHead2', 'LeftFemurHead2', 'RightCrus', 'LeftCrus'):
            subset = [r for r in rows if r['checkpoint'] == mode and float(r['force_N']) == 6 and r['joint'] == joint]
            values = [mean([r[key] for r in subset]) for key in ('pre_torque_pct', 'post_torque_pct', 'target_excursion_deg', 'actual_excursion_deg', 'tracking_error_deg')]
            lines.append('| ' + mode + ' | ' + joint + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    lines += ['', '## New policy direction results', '', '| Force N | Angle degrees | Recovered / 5 | Falls / 5 |', '|---:|---:|---:|---:|']
    for force in (4, 5, 6, 8):
        for angle in (0, 90, 180, 270):
            subset = [r for r in trials if r['checkpoint'] == 'new_best' and float(r['force_N']) == force and float(r['angle_deg']) == angle]
            lines.append(f"| {force} | {angle} | {sum(r['outcome'] != 'failed' for r in subset)} | {sum(r['fell'] == 'True' for r in subset)} |")
    with (folder / 'comparison.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print((folder / 'comparison.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
