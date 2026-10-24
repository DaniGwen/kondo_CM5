"""Create and launch an isolated 160 ms push fine-tuning experiment."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def prepare_run(source, run, settings, script_dir, server):
    if not source.is_file():
        raise FileNotFoundError(source)
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, run / 'reference_model.zip')
    shutil.copy2(source, run / 'kondo_td3_v2.zip')
    snapshots = run / 'source_snapshot'
    snapshots.mkdir()
    for name in ('train_kondo_balance.py', 'shoulder_filter.py',
                 'evaluate_kondo_balance.py', 'recovery_metrics.py',
                 'launch_finetune_balance.py'):
        shutil.copy2(script_dir / name, snapshots / name)
    for name in ('adaptive_evaluation.py', 'adaptive_curriculum.py', 'restart_evaluation.py'):
        shutil.copy2(script_dir / name, snapshots / name)
    shutil.copy2(server, snapshots / server.name)
    for name in ('balance_comparison.csv', 'balance_comparison_stronger.csv'):
        if (script_dir / name).is_file():
            shutil.copy2(script_dir / name, run / name)
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
                    source_model=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    settings=settings, additional_timesteps=int(settings['KONDO_TOTAL_TIMESTEPS']),
                    replay='fresh; no previous replay copied')
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, default=here / 'kondo_td3_v2.zip')
    parser.add_argument('--run-dir', type=Path,
                        default=here / 'runs' / ('finetune_push160_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--steps', type=int, default=500_000)
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--webots', type=Path,
                        default=Path(r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--adaptive', action='store_true', help='Mixed pushes with independent evaluation gates')
    parser.add_argument('--webots-port', type=int, default=1235)
    parser.add_argument('--eval-port', type=int, default=5559)
    parser.add_argument('--resume', action='store_true', help='Resume this existing run and its replay')
    args = parser.parse_args()
    if args.steps < 1 or not 1024 <= args.port <= 65535:
        parser.error('Positive steps and an unprivileged TCP port are required')
    run = args.run_dir.resolve()
    settings = {
        'KONDO_RUN_DIR': str(run), 'KONDO_TOTAL_TIMESTEPS': str(args.steps),
        'KONDO_CONTROLLER_LOG': str(run / 'controller.log'),
        'KONDO_RESET_REPLAY': '1', 'KONDO_WARMUP_STEPS': '5000',
        'KONDO_NOISE_SIGMA': '0.20', 'KONDO_TRAIN_SEED': '123',
        'KONDO_RANDOM_SEED': '12345', 'KONDO_WEBOTS_HOST': '127.0.0.1',
        'KONDO_WEBOTS_PORT': str(args.port), 'KONDO_PUSH_DURATION_STEPS': '10',
        'KONDO_PUSH_MIN_FORCE': '4', 'KONDO_PUSH_MAX_FORCE_INITIAL': '4',
        'KONDO_PUSH_MAX_FORCE_FINAL': '6', 'KONDO_PUSH_RAMP_EPISODES': '30',
        'KONDO_NO_PUSH_EPISODES': '3', 'KONDO_PUSH_SETTLE_STEPS': '125',
        'KONDO_PUSH_PROBABILITY': '0.02', 'KONDO_PUSH_COOLDOWN_STEPS': '125',
        'KONDO_RECOVERY_TILT_DEG': '2', 'KONDO_RECOVERY_STABLE_STEPS': '10',
        'KONDO_RECOVERY_TIMEOUT_STEPS': '125', 'KONDO_FALL_TILT_RAD': '0.85',
        'KONDO_FALL_HEIGHT_M': '0.13',
    }
    if args.adaptive:
        settings.update(KONDO_CURRICULUM_FILE=str(run / 'curriculum.json'),
                        KONDO_EVAL_PORT=str(args.eval_port),
                        KONDO_LEARNING_RATE='0.0001', KONDO_NOISE_SIGMA='0.10',
                        KONDO_PUSH_MIN_FORCE='0.5', KONDO_PUSH_MAX_FORCE_INITIAL='4',
                        KONDO_PUSH_MAX_FORCE_FINAL='6', KONDO_NO_PUSH_EPISODES='0',
                        KONDO_EVAL_WEBOTS_PORT=str(args.webots_port + 1))
    if args.resume:
        import zipfile
        if not (run / 'manifest.json').is_file() or not (run / 'kondo_td3_v2_replay_buffer.pkl').is_file():
            parser.error('Resume requires an existing run manifest, model and replay buffer')
        if (run / 'STOP').exists():
            parser.error('Remove the STOP file before resuming')
        with zipfile.ZipFile(run / 'reference_model.zip') as archive:
            initial_steps = json.loads(archive.read('data'))['num_timesteps']
        with zipfile.ZipFile(run / 'kondo_td3_v2.zip') as archive:
            current_steps = json.loads(archive.read('data'))['num_timesteps']
        remaining = args.steps - (current_steps - initial_steps)
        if remaining <= 0:
            parser.error('Run has already reached the requested additional steps')
        settings.update(KONDO_TOTAL_TIMESTEPS=str(remaining), KONDO_RESET_REPLAY='0')
    if not args.prepare_only:
        if not args.webots.is_file():
            parser.error(f'Webots not found: {args.webots}')
        subprocess.run([sys.executable, '-c',
                        'import numpy, gymnasium, stable_baselines3, torch, tensorboard'], check=True)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        if args.adaptive:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', args.eval_port))
    server = root / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py'
    if args.resume:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        (run / f'resume_{stamp}.json').write_text(json.dumps(dict(settings=settings,
            completed_steps=current_steps - initial_steps, remaining_steps=remaining), indent=2), encoding='utf-8')
        snapshots = run / f'resume_source_{stamp}'
        snapshots.mkdir()
        for name in ('training.log', 'status.json'):
            if (run / name).exists():
                shutil.copy2(run / name, snapshots / name)
        for name in ('train_kondo_balance.py', 'adaptive_evaluation.py',
                     'adaptive_curriculum.py', 'restart_evaluation.py', 'launch_finetune_balance.py'):
            shutil.copy2(here / name, snapshots / name)
    else:
        prepare_run(args.source_model.resolve(), run, settings, here, server)
    if args.adaptive and not args.resume:
        (run / 'curriculum.json').write_text(json.dumps(dict(stage=0, max_force_N=4)), encoding='utf-8')
    print(f'Prepared {run}', flush=True)
    if args.prepare_only:
        return
    env = os.environ.copy()
    env.update(settings)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    world = root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
    started = time.time()
    status_path = run / 'status.json'
    status = dict(state='starting', run_dir=str(run), launcher_pid=os.getpid())
    if args.resume:
        status.update(resumed=True, completed_steps_before_resume=current_steps - initial_steps,
                      remaining_steps=remaining)
    def save_status():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    save_status()
    try:
        with (run / 'webots.log').open('w', encoding='utf-8') as sim_log:
            sim = subprocess.Popen([str(args.webots), '--batch', '--mode=fast',
                                    '--no-rendering', '--minimize', '--stdout', '--stderr',
                                    f'--port={args.webots_port}', str(world)],
                                   cwd=root, env=env, stdout=sim_log, stderr=subprocess.STDOUT)
            status['webots_launcher_pid'] = sim.pid
            save_status()
            # Do not connect to probe readiness: the controller accepts one client.
            while time.time() - started < 120:
                controller_log = run / 'controller.log'
                log = controller_log.read_text(encoding='utf-8', errors='replace') if controller_log.exists() else ''
                if f'Listening on 0.0.0.0:{args.port}' in log:
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f'Controller did not become ready; inspect {run / "controller.log"} and webots.log')
            if args.adaptive:
                eval_env = env.copy()
                eval_env.pop('KONDO_CURRICULUM_FILE', None)
                eval_env['KONDO_WEBOTS_PORT'] = str(args.eval_port)
                eval_env['KONDO_CONTROLLER_LOG'] = str(run / 'evaluation_controller.log')
                eval_log = (run / 'evaluation_webots.log').open('w', encoding='utf-8')
                eval_sim = subprocess.Popen([str(args.webots), '--batch', '--mode=fast',
                                             '--no-rendering', '--minimize', '--stdout', '--stderr',
                                             f'--port={args.webots_port + 1}', str(world)],
                                            cwd=root, env=eval_env, stdout=eval_log, stderr=subprocess.STDOUT)
                status['evaluation_webots_pid'] = eval_sim.pid
                save_status()
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    log_path = run / 'evaluation_controller.log'
                    if log_path.exists() and f'Listening on 0.0.0.0:{args.eval_port}' in log_path.read_text(encoding='utf-8', errors='replace'):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Evaluation simulator startup timed out')
            with (run / 'training.log').open('w', encoding='utf-8') as train_log:
                trainer = subprocess.Popen([sys.executable, '-u', str(here / 'train_kondo_balance.py')],
                                           cwd=here, env=env, stdout=train_log, stderr=subprocess.STDOUT)
                status.update(state='training', trainer_pid=trainer.pid)
                save_status()
                code = trainer.wait()
                status.update(state='completed' if code == 0 else 'failed', exit_code=code,
                              elapsed_seconds=time.time() - started)
                save_status()
                if code:
                    raise RuntimeError(f'Training exited {code}; inspect {run / "training.log"}')
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save_status()
        raise
    print(f'Finished. Model and replay saved in {run}', flush=True)


if __name__ == '__main__':
    main()
