"""Isolate neutral and policy-induced hip loading with no external pushes."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', choices=('normal', 'no_self_collision', 'friction_half', 'box_slab'), default='normal')
    parser.add_argument('--torque-scale', type=float, default=1)
    parser.add_argument('--best-model', type=Path)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    suffix = '' if args.ablation == 'normal' else '_' + args.ablation
    if args.torque_scale != 1:
        suffix += '_torque' + str(args.torque_scale).replace('.', 'p')
    suffix += args.tag
    output = here / ('runs/hip_load_20261023_01' + suffix)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 5587))
    output.mkdir(exist_ok=False)
    from diagnose_joints import sensor_world
    source = root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
    world = source.with_name('hip_load_20261023_01' + suffix + '.wbt')
    content = sensor_world(source.read_text(encoding='utf-8'))
    if args.ablation == 'box_slab':
        from box_slab_world import use_box_slab
        content = use_box_slab(content)
    if args.torque_scale != 1:
        from box_slab_world import scale_leg_torque
        content = scale_leg_torque(content, args.torque_scale)
    if args.ablation == 'no_self_collision':
        assert content.count('selfCollision TRUE') == 1
        content = content.replace('selfCollision TRUE', 'selfCollision FALSE')
    if args.ablation == 'friction_half':
        marker = 'contactProperties ['
        assert content.count(marker) == 1
        content = content.replace(marker, marker + '\n ContactProperties { material1 "default" material2 "default" coulombFriction [0.5] }', 1)
    world.write_text(content, encoding='utf-8')
    os.environ.update(KONDO_RESET_JOINTS='1', KONDO_WEBOTS_PORT='5587',
        KONDO_JOINT_DIAGNOSTICS=str(output / 'joints.csv'),
        KONDO_STANCE_DIAGNOSTICS=str(output / 'contacts.csv'),
        KONDO_CONTROLLER_LOG=str(output / 'controller.log'),
        KONDO_FALL_TILT_RAD='.85', KONDO_FALL_HEIGHT_M='.13')
    os.environ.pop('KONDO_CURRICULUM_FILE', None)
    status = dict(state='starting', completed_trials=0, total_trials=12, ablation=args.ablation, torque_scale=args.torque_scale)
    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save()
    env = None
    try:
        with (output / 'webots.log').open('w') as log:
            subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
                '--port=1269', str(world)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            log = output / 'controller.log'
            if log.exists() and 'Listening on 0.0.0.0:5587' in log.read_text(encoding='utf-8'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Simulator startup timed out')
        import numpy as np
        from stable_baselines3 import SAC
        from train_kondo_balance import KondoBalanceEnv
        run = here / 'runs/sac_clean_reset_20261023_01'
        initial = SAC.load(run / 'reference_model.zip', device='cpu')
        best = SAC.load(args.best_model or run / 'best_model.zip', device='cpu')
        env = KondoBalanceEnv(port=5587)
        with (output / 'body.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['episode', 'mode', 'repeat', 'step', 'tilt_deg', 'height_m', 'fell'])
            episode = 0
            for repeat in range(1, 4):
                for mode in ('neutral', 'initial', 'best', 'best_then_neutral'):
                    episode += 1
                    status.update(state='evaluating', mode=mode, repeat=repeat)
                    save()
                    # Zero force, push deferred beyond the hold interval: deterministic reset, no disturbance.
                    obs, _ = env.reset(options={'push_trial': (0, 0, 10, 2000)})
                    for step in range(1, 626):
                        policy = initial if mode == 'initial' else best
                        neutral = mode == 'neutral' or (mode == 'best_then_neutral' and step > 125)
                        action = np.zeros(14, dtype=np.float32) if neutral else policy.predict(obs, deterministic=True)[0]
                        obs, _, terminated, truncated, info = env.step(action)
                        writer.writerow([episode, mode, repeat, step, float(np.degrees(info['tilt_rad'])), info['height_m'], info['fell']])
                        if terminated or truncated:
                            break
                    stream.flush()
                    status['completed_trials'] = episode
                    save()
        status['state'] = 'completed'
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
