"""Record requested, filtered and measured joint behavior in matched push tests."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

JOINTS = ['RightFemurHead1', 'RightFemurHead2', 'RightFemur', 'RightCrus', 'RightAnkle', 'RightFoot',
          'LeftFemurHead1', 'LeftFemurHead2', 'LeftFemur', 'LeftCrus', 'LeftAnkle', 'LeftFoot',
          'RightShoulder', 'LeftShoulder']


def sensor_world(source):
    found = []
    def add(match):
        name = match.group(2)
        if name not in JOINTS:
            return match.group(0)
        found.append(name)
        return match.group(0) + '\n PositionSensor { name "diag_' + name + '" }'
    result = re.sub(r'(RotationalMotor\s*\{\s*name\s+"([^"]+)"[^}]*\})', add, source)
    if sorted(found) != sorted(JOINTS):
        raise ValueError('Could not identify exactly one motor for each policy joint')
    return result


def merge_and_summarize(output):
    import numpy as np
    with (output / 'commands.csv').open(newline='') as stream:
        commands = {(r['episode'], r['step'], r['joint']): r for r in csv.DictReader(stream)}
    groups = {}
    with (output / 'measured_joints.csv').open(newline='') as source, (output / 'joint_traces.csv').open('w', newline='') as destination:
        writer = None
        for measured in csv.DictReader(source):
            command = commands.pop((measured['episode'], measured['step'], measured['joint']), None)
            if command is None:
                raise ValueError('Measured sample has no matching command')
            row = dict(command, **{k: v for k, v in measured.items() if k not in command})
            actual = float(measured['actual_rad'])
            target = float(row['filtered_rad'])
            row['tracking_error_rad'] = target - actual
            if writer is None:
                writer = csv.DictWriter(destination, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            groups.setdefault((row['mode'], row['force_N'], row['angle_deg'], row['joint']), []).append(row)
    if commands:
        raise ValueError('Some commands have no measured sample')
    summaries = []
    for (mode, force, angle, joint), rows in groups.items():
        before = [r for r in rows if int(r['step']) < 125][-25:]
        after = [r for r in rows if int(r['step']) >= 125]
        if not after:
            continue
        def array(name, subset=after):
            return np.array([float(r[name]) for r in subset])
        actual = array('actual_rad')
        filtered = array('filtered_rad')
        raw = array('requested_rad')
        baseline = float(np.mean(array('actual_rad', before))) if before else float(actual[0])
        limit = .2 if 'Shoulder' in joint else .1
        summaries.append(dict(mode=mode, force_N=force, angle_deg=angle, joint=joint,
            measured_motion_from_pre_push_deg=float(np.degrees(np.max(np.abs(actual - baseline)))),
            target_motion_from_pre_push_deg=float(np.degrees(np.max(np.abs(filtered - np.mean(array('filtered_rad', before)))))) if before else 0.,
            requested_range_deg=float(np.degrees(np.ptp(raw))),
            filtered_range_deg=float(np.degrees(np.ptp(filtered))),
            mean_tracking_error_deg=float(np.degrees(np.mean(np.abs(filtered - actual)))),
            max_tracking_error_deg=float(np.degrees(np.max(np.abs(filtered - actual)))),
            requested_at_limit_fraction=float(np.mean(np.abs(raw) >= .95 * limit)),
            filtered_at_limit_fraction=float(np.mean(np.abs(filtered) >= .95 * limit)),
            torque_near_limit_fraction=float(np.mean(np.abs(array('torque_Nm')) >= .95 * array('max_torque_Nm')))))
    with (output / 'joint_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=here / 'runs' / ('joint_diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--port', type=int, default=5564)
    parser.add_argument('--webots-port', type=int, default=1246)
    parser.add_argument('--no-self-collision', action='store_true', help='Diagnostic ablation only')
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'td3', 'sac'), default=['baseline', 'td3', 'sac'])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    sac_path = here / 'runs/sac_balance_01/kondo_sac.zip'
    td3_path = here / 'runs/finetune_push160_20261021_02/reference_model.zip'
    for path in (sac_path, td3_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    output.mkdir(parents=True, exist_ok=False)
    world_source = root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
    world = world_source.with_name('joint_diagnostics_' + output.name + '.wbt')
    with world.open('x', encoding='utf-8') as stream:
        content = sensor_world(world_source.read_text(encoding='utf-8'))
        if args.no_self_collision:
            if content.count('selfCollision TRUE') != 1:
                raise ValueError('Expected one explicit selfCollision TRUE setting')
            content = content.replace('selfCollision TRUE', 'selfCollision FALSE')
        stream.write(content)
    env_settings = os.environ.copy()
    env_settings.pop('KONDO_CURRICULUM_FILE', None)
    env_settings.update(KONDO_WEBOTS_PORT=str(args.port), KONDO_JOINT_DIAGNOSTICS=str(output / 'measured_joints.csv'),
                        KONDO_CONTROLLER_LOG=str(output / 'controller.log'), KONDO_PUSH_DURATION_STEPS='10',
                        KONDO_PUSH_MIN_FORCE='0', KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='4',
                        KONDO_RECOVERY_TIMEOUT_STEPS='125', KONDO_RANDOM_SEED='12345')
    status = dict(state='starting', output_dir=str(output), world=str(world), completed_trials=0,
                  total_trials=12 * len(args.modes), self_collision=not args.no_self_collision)
    def save_status():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save_status()
    env = None
    try:
        with (output / 'webots.log').open('w') as log:
            simulator = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
                f'--port={args.webots_port}', str(world)], cwd=root, env=env_settings, stdout=log, stderr=subprocess.STDOUT)
            status['webots_launcher_pid'] = simulator.pid
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                controller_log = output / 'controller.log'
                if controller_log.exists() and f'Listening on 0.0.0.0:{args.port}' in controller_log.read_text(encoding='utf-8'):
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Diagnostic controller startup timed out')
        import numpy as np
        from stable_baselines3 import SAC, TD3
        from train_kondo_balance import KondoBalanceEnv
        from recovery_metrics import RecoveryTracker
        env = KondoBalanceEnv(port=args.port)
        models = [(mode, None if mode == 'baseline' else
                   TD3.load(td3_path, device='cpu') if mode == 'td3' else SAC.load(sac_path, device='cpu'))
                  for mode in args.modes]
        status['state'] = 'evaluating'
        with (output / 'commands.csv').open('w', newline='') as stream, (output / 'outcomes.csv').open('w', newline='') as outcomes:
            writer = csv.DictWriter(stream, fieldnames=['episode', 'step', 'mode', 'force_N', 'angle_deg', 'joint',
                'requested_rad', 'filtered_rad', 'tilt_deg', 'fell'])
            writer.writeheader()
            outcome_writer = csv.DictWriter(outcomes, fieldnames=['episode', 'mode', 'force_N', 'angle_deg', 'outcome', 'fell'])
            outcome_writer.writeheader()
            episode = 0
            for mode, model in models:
                for force in (4, 6, 8):
                    for angle in (0, 90, 180, 270):
                        episode += 1
                        obs, _ = env.reset(options={'push_trial': (force, angle, 10, 125)})
                        tracker = RecoveryTracker()
                        for step in range(1, 250):
                            action = np.zeros(14, dtype=np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                            obs, _, terminated, truncated, info = env.step(action)
                            for i, joint in enumerate(JOINTS):
                                writer.writerow(dict(episode=episode, step=step, mode=mode, force_N=force, angle_deg=angle,
                                    joint=joint, requested_rad=float(action[i]), filtered_rad=float(env.previous_applied_action[i]),
                                    tilt_deg=float(np.degrees(info['tilt_rad'])), fell=info['fell']))
                            if step >= 125:
                                tracker.update(float(np.degrees(info['tilt_rad'])), step - 124,
                                               pushing=step - 124 < 10, fell=info['fell'])
                            if terminated or truncated:
                                break
                        outcome_writer.writerow(dict(episode=episode, mode=mode, force_N=force, angle_deg=angle,
                                                     outcome=tracker.outcome or 'failed', fell=info['fell']))
                        stream.flush()
                        outcomes.flush()
                        status['completed_trials'] = episode
                        save_status()
        env.close()
        env = None
        merge_and_summarize(output)
        status['state'] = 'completed'
        save_status()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save_status()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
