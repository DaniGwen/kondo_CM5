"""Matched neutral/TD3/SAC evaluation with clean joint initialization."""
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    output = here / 'runs/clean_reset_evaluation_20261023_01'
    port = 5580
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(KONDO_WEBOTS_PORT=str(port), KONDO_RESET_JOINTS='1',
        KONDO_WEBOTS_HOST='127.0.0.1', KONDO_CONTROLLER_LOG=str(output / 'controller.log'),
        KONDO_PUSH_MIN_FORCE='0', KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='4',
        KONDO_PUSH_DURATION_STEPS='10', KONDO_RECOVERY_TILT_DEG='2', KONDO_RECOVERY_STABLE_STEPS='10',
        KONDO_RECOVERY_TIMEOUT_STEPS='125', KONDO_FALL_TILT_RAD='.85', KONDO_FALL_HEIGHT_M='.13')
    os.environ.update(settings)
    os.environ.pop('KONDO_CURRICULUM_FILE', None)
    status = dict(state='starting', completed_models=0, total_models=2, clean_reset=True)
    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save()
    env = None
    try:
        with (output / 'webots.log').open('w') as log:
            sim = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
                '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr', '--port=1262',
                str(root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt')], cwd=root, env=os.environ.copy(),
                stdout=log, stderr=subprocess.STDOUT)
        status['webots_launcher_pid'] = sim.pid
        save()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            log = output / 'controller.log'
            if log.exists() and f'Listening on 0.0.0.0:{port}' in log.read_text(encoding='utf-8'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Evaluation simulator startup timed out')
        from stable_baselines3 import TD3, SAC
        from train_kondo_balance import KondoBalanceEnv
        from evaluate_kondo_balance import compare
        env = KondoBalanceEnv(port=port)
        summaries = []
        for algorithm, cls, path in [
                ('td3', TD3, here / 'runs/finetune_push160_20261021_02/reference_model.zip'),
                ('sac', SAC, here / 'runs/sac_balance_01/kondo_sac.zip')]:
            status.update(state='evaluating', model=algorithm)
            save()
            model = cls.load(path, device='cpu')
            args = SimpleNamespace(algorithm=algorithm, episodes=3, forces=[4., 6., 8.],
                angles=[0., 90., 180., 270.], durations=[160.], push_step=125,
                tilt_threshold=2., stable_steps=10, timeout_steps=125, output=str(output / f'{algorithm}.csv'))
            compare(env, model, args)
            with Path(args.output).open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            for mode in ('baseline', algorithm):
                for force in ('4.0', '6.0', '8.0'):
                    subset = [r for r in rows if r['mode'] == mode and r['force_N'] == force]
                    summaries.append(dict(evaluation=algorithm, mode=mode, force_N=force, trials=len(subset),
                        recovered=sum(r['outcome'] == 'recovered' for r in subset),
                        no_measurable_disturbance=sum(r['outcome'] == 'no_measurable_disturbance' for r in subset),
                        failed=sum(r['outcome'] == 'failed' for r in subset),
                        falls=sum(r['fell'] == 'True' for r in subset)))
            with (output / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
            status['completed_models'] += 1
            save()
        status['state'] = 'completed'
        save()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise
    finally:
        if env:
            env.close()


if __name__ == '__main__':
    main()
