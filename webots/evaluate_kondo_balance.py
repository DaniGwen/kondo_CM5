"""Deterministic evaluation of the saved Kondo TD3 actor against one Webots server.

Start the Webots controller first. Stop train_kondo_balance.py before running.
This does NOT train or overwrite the model. The imported environment applies
exactly the same shoulder deadband, hysteresis, EMA and rate limits as v3 training.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from stable_baselines3 import TD3, SAC
from train_kondo_balance import KondoBalanceEnv
from recovery_metrics import RecoveryTracker


def compare(env, model, args):
    """Each reset requests exactly one scheduled push, bypassing curriculum/RNG."""
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for repeat in range(args.episodes):
            for force in args.forces:
                for duration in args.durations:
                    duration_steps = int(np.ceil(duration / 16))
                    for angle in args.angles:
                        # Alternate order to reduce systematic order effects.
                        policy_mode = getattr(args, 'algorithm', 'td3')
                        modes = ('baseline', policy_mode) if repeat % 2 == 0 else (policy_mode, 'baseline')
                        for mode in modes:
                            obs, _ = env.reset(options={'push_trial': (
                                force, angle, duration_steps, args.push_step)})
                            tracker = RecoveryTracker(args.tilt_threshold, args.stable_steps, args.timeout_steps)
                            fell = False
                            command_sum = 0.0
                            for step in range(1, args.push_step + args.timeout_steps):
                                action = (np.zeros(env.action_space.shape, dtype=np.float32)
                                          if mode == 'baseline' else model.predict(obs, deterministic=True)[0])
                                obs, _, terminated, truncated, info = env.step(action)
                                command_sum += float(np.linalg.norm(env.previous_applied_action))
                                fell = info['fell']
                                if step >= args.push_step:
                                    age = step - args.push_step + 1
                                    tracker.update(float(np.degrees(info['tilt_rad'])), age,
                                                   pushing=age < duration_steps, fell=fell)
                                if terminated or truncated or tracker.outcome is not None:
                                    break
                            row = dict(repeat=repeat + 1, mode=mode, force_N=force,
                                       angle_deg=angle, duration_ms=duration_steps * 16,
                                       impulse_Ns=force * duration_steps * .016,
                                       outcome=tracker.outcome or 'failed', fell=fell,
                                       peak_tilt_deg=tracker.peak,
                                       outcome_time_s=tracker.elapsed_steps * .016,
                                       mean_command_norm_rad=command_sum / step,
                                       reason='pre_push_fall' if step < args.push_step else 'trial')
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            stream.flush()
                            rows.append(row)
                            print(row, flush=True)
    for mode in ('baseline', getattr(args, 'algorithm', 'td3')):
        counts = Counter(r['outcome'] for r in rows if r['mode'] == mode)
        pre_push = sum(r['reason'] == 'pre_push_fall' for r in rows if r['mode'] == mode)
        measurable = counts['recovered'] + counts['failed'] - pre_push
        rate = f"{counts['recovered'] / measurable:.1%}" if measurable else 'N/A'
        print(f'{mode}: {dict(counts)}, pre_push_falls={pre_push}, recovery_rate_measurable={rate}')
    print(f'Results saved to {output.resolve()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='kondo_td3_v2.zip')
    parser.add_argument('--algorithm', choices=('td3', 'sac'), default='td3')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--mode', choices=('td3', 'sac', 'baseline', 'compare'), default='td3')
    parser.add_argument('--forces', nargs='+', type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument('--durations', nargs='+', type=float, default=[80, 160], help='milliseconds, rounded up to 16 ms')
    parser.add_argument('--angles', nargs='+', type=float, default=[0, 90, 180, 270])
    parser.add_argument('--push-step', type=int, default=125)
    parser.add_argument('--timeout-steps', type=int, default=125)
    parser.add_argument('--stable-steps', type=int, default=10)
    parser.add_argument('--tilt-threshold', type=float, default=2.0)
    parser.add_argument('--output', default='balance_comparison.csv')
    args = parser.parse_args()
    if args.mode == 'sac':
        args.algorithm = 'sac'
    if args.mode in ('td3', 'sac'):
        args.mode = args.algorithm
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if (not all(np.isfinite(v) for v in args.forces + args.durations + args.angles) or
            min(args.forces) < 0 or max(args.forces) > 100 or min(args.durations) <= 0 or
            args.timeout_steps < 1 or args.stable_steps < 1 or
            not np.isfinite(args.tilt_threshold) or args.tilt_threshold <= 0 or
            args.push_step < 1 or args.push_step + args.timeout_steps > 4000 or
            np.ceil(max(args.durations) / 16) >= args.timeout_steps):
        parser.error('Invalid push settings or trial exceeds episode length')
    if args.mode != 'baseline' and not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')

    algorithm = SAC if args.algorithm == 'sac' else TD3
    model = algorithm.load(args.model, device='cpu') if args.mode != 'baseline' else None
    env = KondoBalanceEnv()
    try:
        if args.mode == 'compare':
            compare(env, model, args)
            return
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            last_info = {}
            action_norm_sum = 0.0
            steps = 0
            while not done:
                if model is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, last_info = env.step(action)
                steps += 1
                action_norm_sum += float(np.linalg.norm(env.previous_applied_action))
                done = terminated or truncated
            print(f'[{args.mode.upper()}] episode={episode} steps={steps} '
                  f'standing_success={last_info.get("standing_success")} fell={last_info.get("fell")} '
                  f'peak_tilt={np.degrees(last_info.get("max_tilt_rad", 0.0)):.2f}deg '
                  f'mean_command_norm={action_norm_sum / max(1, steps):.5f}rad', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
