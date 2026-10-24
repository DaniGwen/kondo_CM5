"""Summarize repeated trials and measured joint tracking without simulator access."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent / 'runs/sac_checkpoint_comparison_20261023_01'
    assert json.loads((folder / 'status.json').read_text())['state'] == 'completed'
    trials = list(csv.DictReader((folder / 'trials.csv').open(newline='')))
    mapping = {r['episode']: r for r in trials}
    groups = defaultdict(list)
    with (folder / 'measured_joints.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            meta = mapping[row['episode']]
            # Keep each trial separate; summarize before and after the push.
            groups[(row['episode'], row['joint'])].append(row)
    joint_rows = []
    for (episode, joint), rows in groups.items():
        before = [r for r in rows if 100 <= int(r['step']) < 125]
        after = [r for r in rows if int(r['step']) >= 125]
        if not before or not after:
            continue
        meta = mapping[episode]
        mean = lambda seq: sum(seq) / len(seq)
        baseline = mean([float(r['actual_rad']) for r in before])
        target = mean([float(r['server_target_rad']) for r in before])
        joint_rows.append(dict(episode=episode, checkpoint=meta['checkpoint'], force_N=meta['force_N'],
            angle_deg=meta['angle_deg'], repeat=meta['repeat'], joint=joint,
            actual_excursion_deg=math.degrees(max(abs(float(r['actual_rad']) - baseline) for r in after)),
            target_excursion_deg=math.degrees(max(abs(float(r['server_target_rad']) - target) for r in after)),
            tracking_error_deg=math.degrees(mean([abs(float(r['actual_rad']) - float(r['server_target_rad'])) for r in after])),
            torque_limit_fraction=mean([float(abs(float(r['torque_Nm'])) >= .95 * float(r['max_torque_Nm'])) for r in after]),
            pre_push_torque_limit_fraction=mean([float(abs(float(r['torque_Nm'])) >= .95 * float(r['max_torque_Nm'])) for r in before])))
    with (folder / 'joint_trial_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)
    lines = ['', '## Direction consistency', '', '| Checkpoint | Force N | Angle degrees | Passed / 5 | Falls / 5 |', '|---|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['checkpoint'] for r in trials):
        for force in ('4', '6', '8'):
            for angle in ('0', '90', '180', '270'):
                subset = [r for r in trials if (r['checkpoint'], r['force_N'], r['angle_deg']) == (name, force, angle)]
                lines.append(f"| {name} | {force} | {angle} | {sum(r['outcome'] != 'failed' for r in subset)} | {sum(r['fell'] == 'True' for r in subset)} |")
    lines += ['', '## Joint response at 6 N', '',
              'Means across all twenty 6 N trials per checkpoint. Excursion is maximum change from each trial pre-push mean. Torque percentages are fractions of sampled steps at >=95% motor torque limit. Actual movement includes passive displacement; it is not proof of active compensation.', '',
              '| Checkpoint | Joint | Target excursion deg | Actual excursion deg | Tracking error deg | Torque limit % | Pre-push torque limit % |',
              '|---|---|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['checkpoint'] for r in trials):
        for joint in dict.fromkeys(r['joint'] for r in joint_rows):
            if not any(part in joint for part in ('Femur', 'Crus')):
                continue
            subset = [r for r in joint_rows if r['checkpoint'] == name and r['force_N'] == '6' and r['joint'] == joint]
            values = [sum(r[key] for r in subset) / len(subset) for key in ('target_excursion_deg', 'actual_excursion_deg', 'tracking_error_deg', 'torque_limit_fraction', 'pre_push_torque_limit_fraction')]
            lines.append(f'| {name} | {joint} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {100*values[3]:.1f} | {100*values[4]:.1f} |')
    with (folder / 'comparison.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print((folder / 'comparison.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
