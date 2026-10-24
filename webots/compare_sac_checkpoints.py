"""Five matched repetitions per checkpoint, with measured joint telemetry."""
import csv
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    run = here / 'runs/sac_clean_reset_20261023_01'
    parser = argparse.ArgumentParser()
    parser.add_argument('--friction', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--box-slab', action='store_true')
    parser.add_argument('--torque-scale', type=float, default=1)
    parser.add_argument('--new-best', type=Path)
    parser.add_argument('--forces', nargs='+', type=float, default=[4, 6, 8])
    parser.add_argument('--v2-run', type=Path)
    args = parser.parse_args()
    if args.friction is not None and not 0 < args.friction <= 1:
        parser.error('Friction must be above zero and at most 1')
    output = args.output.resolve() if args.output else here / 'runs/sac_checkpoint_comparison_20261023_01'
    models = [('initial', run / 'reference_model.zip'),
              ('best_188227', run / 'best_model.zip'),
              ('step_488227', run / 'checkpoints_sac/kondo_sac_488227_steps.zip'),
              ('final_500000', run / 'kondo_sac.zip')]
    if args.friction is not None:
        models = [('neutral', None), ('best_188227', run / 'best_model.zip')]
    if args.new_best:
        models = [('neutral', None), ('previous_best', run / 'best_model.zip'), ('new_best', args.new_best.resolve())]
    if args.v2_run:
        os.environ['KONDO_OBSERVATION_VERSION'] = '2'
        models = [('neutral', None), ('best_v2', args.v2_run / 'best_model.zip'), ('latest_v2', args.v2_run / 'kondo_sac.zip')]
    for _, path in models:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 5586))
    output.mkdir(exist_ok=False)
    from diagnose_joints import sensor_world
    source = root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
    world = source.with_name(output.name + '.wbt')
    content = sensor_world(source.read_text(encoding='utf-8'))
    if args.box_slab:
        from box_slab_world import use_box_slab
        content = use_box_slab(content)
    if args.torque_scale != 1:
        from box_slab_world import scale_leg_torque
        content = scale_leg_torque(content, args.torque_scale)
    if args.friction is not None:
        marker = 'contactProperties ['
        assert content.count(marker) == 1
        content = content.replace(marker, marker + '\n ContactProperties { material1 "default" material2 "default" coulombFriction [' + str(args.friction) + '] }', 1)
    world.write_text(content, encoding='utf-8')
    settings = dict(KONDO_RESET_JOINTS='1', KONDO_WEBOTS_PORT='5586',
                    KONDO_WEBOTS_HOST='127.0.0.1',
                    KONDO_JOINT_DIAGNOSTICS=str(output / 'measured_joints.csv'),
                    KONDO_STANCE_DIAGNOSTICS=str(output / 'contacts.csv'),
                    KONDO_CONTROLLER_LOG=str(output / 'controller.log'),
                    KONDO_RANDOM_SEED='12345', KONDO_FALL_TILT_RAD='.85',
                    KONDO_FALL_HEIGHT_M='.13', KONDO_RECOVERY_TILT_DEG='2',
                    KONDO_RECOVERY_STABLE_STEPS='10', KONDO_RECOVERY_TIMEOUT_STEPS='125')
    os.environ.update(settings)
    os.environ.pop('KONDO_CURRICULUM_FILE', None)
    status = dict(state='starting', completed_trials=0, total_trials=20 * len(args.forces) * len(models), forces=args.forces, friction=args.friction, box_slab=args.box_slab, torque_scale=args.torque_scale,
                  settings=settings, checkpoints={name: str(path) for name, path in models})
    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save()
    env = None
    try:
        with (output / 'webots.log').open('w') as log:
            sim = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
                '--port=1268', str(world)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status['simulator_pid'] = sim.pid
        save()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            log = output / 'controller.log'
            if log.exists() and 'Listening on 0.0.0.0:5586' in log.read_text(encoding='utf-8'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Simulator startup timed out')
        import numpy as np
        from stable_baselines3 import SAC
        from train_kondo_balance import KondoBalanceEnv
        from recovery_metrics import RecoveryTracker
        env = KondoBalanceEnv(port=5586)
        policies = {name: SAC.load(path, device='cpu') if path is not None else None for name, path in models}
        results = []
        with (output / 'trials.csv').open('w', newline='') as trials, (output / 'body.csv').open('w', newline='') as body:
            writer = csv.DictWriter(trials, fieldnames=['episode', 'checkpoint', 'repeat', 'force_N', 'angle_deg', 'outcome', 'fell', 'peak_tilt_deg', 'outcome_time_s', 'pre_push_fall'])
            writer.writeheader()
            trace = csv.writer(body)
            trace.writerow(['episode', 'checkpoint', 'step', 'tilt_deg', 'height_m'])
            episode = 0
            # Interleave policies so all checkpoints share each repetition's order.
            for repeat in range(1, 6):
                for name, _ in models:
                    for force in args.forces:
                        for angle in (0, 90, 180, 270):
                            episode += 1
                            status.update(state='evaluating', checkpoint=name, repeat=repeat, force_N=force, angle_deg=angle)
                            save()
                            obs, _ = env.reset(options={'push_trial': (force, angle, 10, 125)})
                            tracker = RecoveryTracker()
                            for step in range(1, 250):
                                action = np.zeros(14, dtype=np.float32) if policies[name] is None else policies[name].predict(obs, deterministic=True)[0]
                                obs, _, terminated, truncated, info = env.step(action)
                                trace.writerow([episode, name, step, float(np.degrees(info['tilt_rad'])), info['height_m']])
                                if step >= 125:
                                    age = step - 124
                                    tracker.update(float(np.degrees(info['tilt_rad'])), age, pushing=age < 10, fell=info['fell'])
                                if terminated or truncated or tracker.outcome is not None:
                                    break
                            row = dict(episode=episode, checkpoint=name, repeat=repeat, force_N=force, angle_deg=angle,
                                       outcome=tracker.outcome or 'failed', fell=info['fell'], peak_tilt_deg=tracker.peak,
                                       outcome_time_s=tracker.elapsed_steps * .016, pre_push_fall=step < 125)
                            results.append(row)
                            writer.writerow(row)
                            trials.flush()
                            body.flush()
                            status['completed_trials'] = episode
                            save()
        lines = ['# Repeated SAC checkpoint comparison', '',
                 'Five repetitions per condition; clean reset; 160 ms pushes at step 125; recovery below 2 degrees for 10 steps within 2 seconds. Angles are simulator force angles. No training was performed.', '',
                 '| Checkpoint | Force N | Passed / 20 | Falls | Mean peak tilt degrees |',
                 '|---|---:|---:|---:|---:|']
        for name, _ in models:
            for force in args.forces:
                rows = [r for r in results if r['checkpoint'] == name and r['force_N'] == force]
                lines.append(f"| {name} | {force} | {sum(r['outcome'] != 'failed' for r in rows)} | {sum(r['fell'] for r in rows)} | {np.mean([r['peak_tilt_deg'] for r in rows]):.2f} |")
        (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
        status.update(state='completed')
        save()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
