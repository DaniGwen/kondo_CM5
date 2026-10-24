"""Restart a disconnected, dedicated evaluation simulator without touching training."""
import os
from pathlib import Path
import socket
import subprocess
import time


def restart_evaluation(run_dir, port):
    run = Path(run_dir)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    root = Path(__file__).resolve().parents[2]
    log = run / 'evaluation_controller.log'
    offset = log.stat().st_size if log.exists() else 0
    environment = os.environ.copy()
    environment.pop('KONDO_CURRICULUM_FILE', None)
    environment['KONDO_WEBOTS_PORT'] = str(port)
    environment['KONDO_CONTROLLER_LOG'] = str(log)
    with (run / 'evaluation_webots.log').open('a', encoding='utf-8') as output:
        process = subprocess.Popen([
            r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
            '--batch', '--mode=fast', '--no-rendering', '--minimize', '--stdout', '--stderr',
            '--port=' + os.getenv('KONDO_EVAL_WEBOTS_PORT', '1241'),
            environment.get('KONDO_WORLD', str(root / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'))],
            cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(offset)
                if f'Listening on 0.0.0.0:{port}'.encode() in stream.read():
                    from train_kondo_balance import KondoBalanceEnv
                    print(f'[EVALUATION] Restarted simulator launcher PID={process.pid}', flush=True)
                    return KondoBalanceEnv(port=port)
        time.sleep(1)
    raise RuntimeError('Replacement evaluation simulator did not become ready')
