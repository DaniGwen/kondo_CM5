"""Launch a fresh SAC run with clean resets and an independent evaluation server."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime


def start_server(root, run, environment, port, app_port, log_name):
    controller_log = run / log_name
    log_offset = controller_log.stat().st_size if controller_log.exists() else 0
    server_env = environment.copy()
    server_env.update(KONDO_WEBOTS_PORT=str(port), KONDO_CONTROLLER_LOG=str(controller_log))
    if log_name.startswith('evaluation'):
        server_env.pop('KONDO_CURRICULUM_FILE', None)
    with (run / (log_name + '.webots.log')).open('w') as log:
        process = subprocess.Popen([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
            '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
            f'--port={app_port}', environment.get('KONDO_WORLD', str(root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'))],
            cwd=root, env=server_env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if controller_log.exists():
            with controller_log.open('rb') as stream:
                stream.seek(log_offset)
                if f'Listening on 0.0.0.0:{port}'.encode() in stream.read():
                    return process.pid
        time.sleep(1)
    raise RuntimeError('Server startup timed out: ' + str(controller_log))


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--port', type=int, default=5582)
    parser.add_argument('--eval-port', type=int, default=5583)
    parser.add_argument('--webots-port', type=int, default=1264)
    parser.add_argument('--resume', action='store_true', help='Continue saved run toward --steps total')
    parser.add_argument('--world', type=Path, help='World for a fresh run; resume must match the saved world')
    parser.add_argument('--observation-version', choices=('1', '2'), default='1')
    args = parser.parse_args()
    for port in (args.port, args.eval_port):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    run = args.run_dir.resolve()
    environment = os.environ.copy()
    for name in ('KONDO_JOINT_DIAGNOSTICS', 'KONDO_STANCE_DIAGNOSTICS', 'KONDO_RESET_ALL_TARGETS'):
        environment.pop(name, None)
        os.environ.pop(name, None)
    settings = dict(KONDO_OBSERVATION_VERSION=args.observation_version, KONDO_RESET_JOINTS='1', KONDO_WEBOTS_HOST='127.0.0.1',
        KONDO_WEBOTS_PORT=str(args.port), KONDO_EVAL_PORT=str(args.eval_port),
        KONDO_EVAL_WEBOTS_PORT=str(args.webots_port + 1), KONDO_CURRICULUM_FILE=str(run / 'curriculum.json'),
        KONDO_PUSH_MIN_FORCE='.5', KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='6',
        KONDO_PUSH_DURATION_STEPS='10', KONDO_NO_PUSH_EPISODES='0', KONDO_PUSH_SETTLE_STEPS='125',
        KONDO_PUSH_PROBABILITY='.02', KONDO_PUSH_COOLDOWN_STEPS='125', KONDO_RANDOM_SEED='12345',
        KONDO_RECOVERY_TILT_DEG='2', KONDO_RECOVERY_STABLE_STEPS='10', KONDO_RECOVERY_TIMEOUT_STEPS='125',
        KONDO_FALL_TILT_RAD='.85', KONDO_FALL_HEIGHT_M='.13', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    environment.update(settings)
    os.environ.update(environment)
    from train_kondo_balance_sac import parse_args, prepare_run
    remaining = args.steps
    completed = 0
    if args.resume:
        import zipfile
        experiment = json.loads((run / 'experiment.json').read_text())
        if experiment['settings'].get('KONDO_OBSERVATION_VERSION', '1') != args.observation_version:
            raise ValueError('Resume requires the same observation version')
        saved_world = Path(experiment['settings'].get('KONDO_WORLD', str(root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'))).resolve()
        if args.world and args.world.resolve() != saved_world:
            raise ValueError('Resume cannot change the saved simulator world; use a fresh run')
        if experiment['settings'].get('KONDO_RESET_JOINTS') != '1':
            raise ValueError('Expected an existing clean-reset SAC experiment')
        if not (run / 'kondo_sac_replay_buffer.pkl').is_file():
            raise FileNotFoundError('Saved SAC replay required for resume')
        with zipfile.ZipFile(run / 'kondo_sac.zip') as archive:
            completed = json.loads(archive.read('data'))['num_timesteps']
        remaining = args.steps - completed
        if remaining <= 0:
            raise ValueError('Requested total steps already reached')
        stop = run / 'STOP'
        if stop.exists():
            stop.unlink()
    world = saved_world if args.resume else (args.world or root / 'webots/kondo_RL/worlds/kondoKHR3-HV-box-slab-corrected.wbt').resolve()
    if not world.is_file():
        raise FileNotFoundError(world)
    settings['KONDO_WORLD'] = str(world)
    environment['KONDO_WORLD'] = str(world)
    os.environ['KONDO_WORLD'] = str(world)
    training_args = parse_args(['--run-dir', str(run), '--steps', str(remaining), '--port', str(args.port)] +
                               (['--resume'] if args.resume else []))
    prepare_run(training_args)
    if not args.resume:
        (run / 'curriculum.json').write_text(json.dumps(dict(stage=0, max_force_N=4, gate_streak=0)))
        (run / 'experiment.json').write_text(json.dumps(dict(algorithm='SAC', initialization='from scratch',
            fresh_replay=True, settings=settings, evaluation_interval=25000,
            best_model_reference='initial SAC policy, not the earlier TD3/SAC actors'), indent=2))
    snapshots = run / ('resume_' + datetime.now().strftime('%Y%m%d_%H%M%S') if args.resume else 'source_snapshot')
    snapshots.mkdir()
    shutil.copy2(world, snapshots / 'simulation_world.wbt')
    for name in ('launch_sac_clean_reset.py', 'train_kondo_balance_sac.py', 'train_kondo_balance.py',
                 'adaptive_evaluation.py', 'adaptive_curriculum.py', 'restart_evaluation.py', 'shoulder_filter.py'):
        shutil.copy2(here / name, snapshots / name)
    shutil.copy2(here / 'balance_reward_v2.py', snapshots / 'balance_reward_v2.py')
    shutil.copy2(root / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py', snapshots / 'kondo_tcp_server.py')
    evaluation = here / 'runs/clean_reset_evaluation_20261023_01'
    for name in ('summary.csv', 'td3.csv', 'sac.csv'):
        if not args.resume:
            shutil.copy2(evaluation / name, run / ('pretraining_' + name))
    status = dict(state='starting', clean_reset=True, algorithm='SAC', requested_steps=args.steps,
                  launcher_pid=os.getpid(), resumed=args.resume, completed_before_resume=completed,
                  remaining_steps=remaining)
    def save():
        (run / 'launch_status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        status['training_simulator_pid'] = start_server(root, run, environment, args.port, args.webots_port, 'controller.log')
        save()
        status['evaluation_simulator_pid'] = start_server(root, run, environment, args.eval_port, args.webots_port + 1,
                                                          'evaluation_controller.log')
        save()
        with (run / 'training.log').open('a' if args.resume else 'w') as log:
            # prepare_run has created only metadata. The trainer starts fresh;
            # call its train() with the prepared directory in a small worker below.
            command = [sys.executable, '-u', str(here / 'launch_sac_clean_reset.py'), '--worker', str(run), str(remaining), str(args.port)]
            if args.resume:
                command.append('--resume')
            trainer = subprocess.Popen(command, cwd=here, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status.update(state='training', trainer_pid=trainer.pid)
            save()
            code = trainer.wait()
            status.update(state='completed' if code == 0 else 'failed', exit_code=code)
            save()
            if code:
                raise RuntimeError('SAC training exited with code ' + str(code))
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        from train_kondo_balance_sac import parse_args, train, KondoBalanceEnv
        run = Path(sys.argv[2])
        args = parse_args(['--run-dir', str(run), '--steps', sys.argv[3], '--port', sys.argv[4]] +
                          (['--resume'] if '--resume' in sys.argv[5:] else []))
        train(args, KondoBalanceEnv(port=args.port), run)
    else:
        main()
