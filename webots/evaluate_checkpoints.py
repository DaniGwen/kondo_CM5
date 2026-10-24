"""Evaluate all fine-tuning checkpoints against matched pushes in one simulator."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def summarize(label, steps, rows):
    policy = [r for r in rows if r['mode'] == 'td3']
    recovered = [r for r in policy if r['outcome'] == 'recovered']
    return dict(model=label, additional_steps=steps, trials=len(policy),
                falls=sum(r['fell'] == 'True' for r in policy),
                failed=sum(r['outcome'] == 'failed' for r in policy),
                recovered=len(recovered),
                no_measurable_disturbance=sum(r['outcome'] == 'no_measurable_disturbance' for r in policy),
                mean_peak_tilt_deg=sum(float(r['peak_tilt_deg']) for r in policy) / len(policy),
                mean_recovery_time_s=(sum(float(r['outcome_time_s']) for r in recovered) / len(recovered)
                                      if recovered else None))


def main():
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--port', type=int, default=5557)
    parser.add_argument('--checkpoint-glob', default='kondo_td3_v2_*_steps.zip')
    parser.add_argument('--webots-port', type=int, default=1236)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if '/' in args.checkpoint_glob or '\\' in args.checkpoint_glob:
        parser.error('Checkpoint glob must be a filename pattern')
    checkpoints = sorted((run / 'checkpoints_kondo_v2').glob(args.checkpoint_glob),
                         key=lambda p: int(p.stem.split('_')[-2]))
    if not checkpoints or not (run / 'reference_model.zip').is_file():
        parser.error('Reference model and checkpoints are required')
    if args.episodes < 1:
        parser.error('Positive episode count required')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    output = (args.output_dir or run / ('checkpoint_evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Set before importing the environment, whose defaults are read at import time.
    settings = dict(KONDO_WEBOTS_PORT=str(args.port), KONDO_WEBOTS_HOST='127.0.0.1',
                    KONDO_CONTROLLER_LOG=str(output / 'controller.log'),
                    KONDO_RANDOM_SEED='12345', KONDO_PUSH_DURATION_STEPS='10',
                    KONDO_PUSH_MIN_FORCE='0', KONDO_PUSH_MAX_FORCE_INITIAL='4',
                    KONDO_PUSH_MAX_FORCE_FINAL='4', KONDO_RECOVERY_TILT_DEG='2',
                    KONDO_RECOVERY_STABLE_STEPS='10', KONDO_RECOVERY_TIMEOUT_STEPS='125',
                    KONDO_FALL_TILT_RAD='0.85', KONDO_FALL_HEIGHT_M='0.13')
    os.environ.update(settings)
    from stable_baselines3 import TD3
    from train_kondo_balance import KondoBalanceEnv
    from evaluate_kondo_balance import compare
    status = dict(state='starting', launcher_pid=os.getpid(), output_dir=str(output),
                  completed_models=0, total_models=len(checkpoints) + 1,
                  forces_N=[4, 6, 8], angles_deg=[0, 90, 180, 270], duration_ms=160,
                  repetitions=args.episodes, settings=settings)
    def save_status():
        (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    save_status()
    env = None
    try:
        with (output / 'webots.log').open('w', encoding='utf-8') as sim_log:
            sim = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                                    '--batch', '--mode=fast', '--no-rendering', '--minimize',
                                    '--stdout', '--stderr', f'--port={args.webots_port}',
                                    str(root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt')],
                                   cwd=root, env=os.environ.copy(), stdout=sim_log, stderr=subprocess.STDOUT)
            status['webots_launcher_pid'] = sim.pid
            save_status()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                log = output / 'controller.log'
                if log.exists() and f'Listening on 0.0.0.0:{args.port}' in log.read_text(encoding='utf-8', errors='replace'):
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Controller startup timed out; inspect controller.log')
            env = KondoBalanceEnv()
            reference_steps = None
            summaries = []
            for model_path in [run / 'reference_model.zip'] + checkpoints:
                model = TD3.load(str(model_path), device='cpu')
                if reference_steps is None:
                    reference_steps = model.num_timesteps
                additional = model.num_timesteps - reference_steps
                label = 'reference' if additional == 0 else f'checkpoint_{additional:06d}'
                csv_path = output / f'{label}.csv'
                status.update(state='evaluating', current_model=label)
                save_status()
                trial_args = SimpleNamespace(episodes=args.episodes, forces=[4., 6., 8.],
                                             durations=[160.], angles=[0., 90., 180., 270.],
                                             push_step=125, tilt_threshold=2., stable_steps=10,
                                             timeout_steps=125, output=str(csv_path))
                compare(env, model, trial_args)
                with csv_path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                summaries.append(summarize(label, additional, rows))
                with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
                status['completed_models'] += 1
                save_status()
            status.update(state='completed')
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
