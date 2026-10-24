"""Neutral stance sweep; diagnostic only, no pushes or learned policy."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import re
from diagnose_joints import sensor_world


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=here / 'runs' / ('stance_sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--offsets', nargs='+', type=float, default=[-4, -2, 0, 2, 4])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--port', type=int, default=5568)
    parser.add_argument('--webots-port', type=int, default=1250)
    parser.add_argument('--gravity', type=float, default=None, help='Diagnostic ablation only')
    parser.add_argument('--ground-z', type=float, default=None, help='Diagnostic ground translation only')
    parser.add_argument('--friction', type=float, default=None, help='Explicit default/default contact friction')
    parser.add_argument('--reset-joints', action='store_true', help='Reset joint coordinates before settling')
    parser.add_argument('--reset-all-targets', action='store_true', help='Reset all motor targets without moving coordinates')
    args = parser.parse_args()
    if args.repetitions < 1 or any(not math.isfinite(x) or abs(math.radians(x)) > .1 for x in args.offsets):
        parser.error('Offsets must be finite and within the existing +/-0.10 rad action limits')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
    world = original.with_name('diagnostic_' + output.name + '.wbt')
    with world.open('x', encoding='utf-8') as stream:
        content = sensor_world(original.read_text(encoding='utf-8'))
        if args.gravity is not None:
            content = content.replace('WorldInfo {', f'WorldInfo {{\n gravity {args.gravity}', 1)
        if args.ground_z is not None:
            before = 'RectangleArena {\n  translation 0 0 -0.1'
            if before not in content:
                raise ValueError('Expected arena translation not found')
            content = content.replace(before, f'RectangleArena {{\n  translation 0 0 {args.ground_z}', 1)
        if args.friction is not None:
            if not math.isfinite(args.friction) or args.friction < 0:
                raise ValueError('Friction must be finite and nonnegative')
            content = content.replace('material1 "foot"', 'material1 "default"').replace('material2 "floor"', 'material2 "default"')
            content, count = re.subn(r'coulombFriction\s*\[\s*2\s*\]',
                                    f'coulombFriction [ {args.friction} ]', content)
            if count != 1:
                raise ValueError('Expected one contact friction entry')
        stream.write(content)
    environment = os.environ.copy()
    environment.pop('KONDO_CURRICULUM_FILE', None)
    environment.update(KONDO_WEBOTS_PORT=str(args.port), KONDO_JOINT_DIAGNOSTICS=str(output / 'joints.csv'),
        KONDO_STANCE_DIAGNOSTICS=str(output / 'contacts.csv'), KONDO_CONTROLLER_LOG=str(output / 'controller.log'),
        KONDO_PUSH_MIN_FORCE='0', KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='4',
        KONDO_PUSH_DURATION_STEPS='10', KONDO_RECOVERY_TIMEOUT_STEPS='125')
    environment['KONDO_RESET_JOINTS'] = '1' if args.reset_joints else '0'
    environment['KONDO_RESET_ALL_TARGETS'] = '1' if args.reset_all_targets else '0'
    status = dict(state='starting', offsets_deg=args.offsets, repetitions=args.repetitions,
                  completed_trials=0, total_trials=len(args.offsets) * args.repetitions,
                  steps_per_trial=500, scoring_steps=[376, 500], pushes=False)
    status.update(gravity=args.gravity, ground_z=args.ground_z, explicit_friction=args.friction,
                  reset_joints=args.reset_joints)
    status['reset_all_targets'] = args.reset_all_targets
    def save_status():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save_status()
    env = None
    try:
        with (output / 'webots.log').open('w') as log:
            process = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
                f'--port={args.webots_port}', str(world)], cwd=root, env=environment,
                stdout=log, stderr=subprocess.STDOUT)
        status['webots_launcher_pid'] = process.pid
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            log = output / 'controller.log'
            if log.exists() and f'Listening on 0.0.0.0:{args.port}' in log.read_text(encoding='utf-8'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Stance diagnostic controller did not start')
        import numpy as np
        from train_kondo_balance import KondoBalanceEnv
        env = KondoBalanceEnv(port=args.port)
        trials = []
        status['state'] = 'evaluating'
        with (output / 'body.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['episode', 'step', 'offset_deg', 'tilt_deg', 'height_m'])
            writer.writeheader()
            episode = 0
            for repeat in range(args.repetitions):
                offsets = args.offsets if repeat % 2 == 0 else list(reversed(args.offsets))
                for offset in offsets:
                    episode += 1
                    # Scheduled zero-force event lies beyond this trial; bypasses random pushes.
                    env.reset(options={'push_trial': (0, 0, 1, 3999)})
                    action = np.zeros(14, dtype=np.float32)
                    action[[1, 7]] = math.radians(offset)
                    for step in range(1, 501):
                        _, _, terminated, _, info = env.step(action)
                        writer.writerow(dict(episode=episode, step=step, offset_deg=offset,
                            tilt_deg=math.degrees(info['tilt_rad']), height_m=info['height_m']))
                        if terminated:
                            break
                    trials.append(dict(episode=episode, repeat=repeat + 1, offset_deg=offset,
                                       steps=step, fell=info['fell']))
                    status['completed_trials'] = episode
                    save_status()
                    stream.flush()
        env.close()
        env = None
        def read(name):
            with (output / name).open(newline='') as stream:
                return list(csv.DictReader(stream))
        joints, contacts, bodies = read('joints.csv'), read('contacts.csv'), read('body.csv')
        summaries = []
        for trial in trials:
            episode = str(trial['episode'])
            samples = [r for r in joints if r['episode'] == episode and int(r['step']) >= 376
                       and r['joint'] in ('LeftFemurHead2', 'RightFemurHead2')]
            body = [r for r in bodies if r['episode'] == episode and int(r['step']) >= 376]
            summary = dict(trial)
            summary['mean_tilt_deg'] = float(np.mean([float(r['tilt_deg']) for r in body])) if body else None
            for side in ('Left', 'Right'):
                subset = [r for r in samples if r['joint'] == side + 'FemurHead2']
                foot = [r for r in contacts if r['episode'] == episode and r['side'] == side and int(r['step']) >= 376]
                summary[side + '_torque_fraction'] = float(np.mean([abs(float(r['torque_Nm'])) / float(r['max_torque_Nm']) for r in subset])) if subset else None
                summary[side + '_error_deg'] = float(np.mean([math.degrees(abs(float(r['server_target_rad']) - float(r['actual_rad']))) for r in subset])) if subset else None
                summary[side + '_contact_fraction'] = float(np.mean([int(r['contact_count']) > 0 for r in foot])) if foot else None
                summary[side + '_foot_tilt_deg'] = float(np.mean([float(r['foot_tilt_deg']) for r in foot])) if foot else None
            summaries.append(summary)
        with (output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        status['state'] = 'completed'
        save_status()
        print(json.dumps(summaries, indent=2), flush=True)
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save_status()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
