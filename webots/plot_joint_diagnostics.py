"""Plot measured versus requested control for a selected diagnostic push."""
import argparse
import csv
from pathlib import Path
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (args.directory / 'joint_traces.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['force_N'] == '6' and r['angle_deg'] == '0']
    joints = [('RightCrus', 'Right knee'), ('LeftCrus', 'Left knee'),
              ('RightFemur', 'Right hip: Femur'), ('LeftFemurHead2', 'Left hip: FemurHead2')]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for col, mode in enumerate(('td3', 'sac')):
        for line, (joint, label) in enumerate(joints):
            ax = axes[line, col]
            selected = [r for r in rows if r['mode'] == mode and r['joint'] == joint]
            times = [(int(r['step']) - 124) * .016 for r in selected]
            for name, color, style, caption in [('requested_rad', '#9ca3af', ':', 'Requested'),
                    ('filtered_rad', '#2563eb', '--', 'Filtered target'), ('actual_rad', '#e06912', '-', 'Measured')]:
                ax.plot(times, [math.degrees(float(r[name])) for r in selected], color=color,
                        linestyle=style, label=caption, linewidth=1.5)
            ax.axvspan(0, .160, color='#ef4444', alpha=.12)
            ax.axhline(math.degrees(.1), color='#999999', linewidth=.6)
            ax.axhline(-math.degrees(.1), color='#999999', linewidth=.6)
            ax.set_title(f'{mode.upper()} — {label}', fontsize=11)
            ax.set_ylabel('Joint angle (degrees)')
            ax.grid(alpha=.2)
            if line == 0:
                ax.legend(fontsize=8, loc='lower left')
    for ax in axes[-1]:
        ax.set_xlabel('Simulation seconds relative to push onset')
    fig.suptitle('6 N push at 0°, 160 ms: joint targets versus measured motion\nRed band: push; gray lines: ±5.73° policy limits', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(args.directory / 'joint_tracking_6N_0deg.png', dpi=160)
    for mode in ('baseline', 'td3', 'sac'):
        selected = [r for r in rows if r['mode'] == mode and r['joint'] == 'LeftFemurHead2']
        for phase, predicate in [('pre_push', lambda r: 100 <= int(r['step']) < 125),
                                 ('post_push', lambda r: int(r['step']) >= 125)]:
            part = [r for r in selected if predicate(r)]
            near = sum(abs(float(r['torque_Nm'])) >= .95 * float(r['max_torque_Nm']) for r in part) / len(part)
            error = sum(abs(float(r['tracking_error_rad'])) for r in part) / len(part)
            print(mode, phase, 'hip_torque_near_limit', round(near, 3), 'mean_error_deg', round(math.degrees(error), 3))


if __name__ == '__main__':
    main()
