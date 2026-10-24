"""Local-only training dashboard. Run with the project's training Python."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
PENDING = None
DOCS = {'model': 'CORRECTED_MODEL.md', 'stepping': 'TD3_STEPPING_V3.md',
        'observations': 'OBSERVATION_V2.md', 'launcher': 'TRAINING_UI.md', 'new-world': 'CORRECTED_ROBOT.md'}


def validate_config(data):
    name = data.get('name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Run name: use 1–64 letters, numbers, underscores or hyphens.')
    if data.get('policy') not in ('TD3', 'SAC'):
        raise ValueError('Select TD3 or SAC.')
    steps = data.get('steps')
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1000 <= steps <= 20000000:
        raise ValueError('Steps must be a whole number from 1,000 to 20,000,000.')
    rate = float(data.get('learning_rate', 0))
    if not math.isfinite(rate) or not 1e-6 <= rate <= .01:
        raise ValueError('Learning rate must be between 0.000001 and 0.01.')
    rewards = {}
    for key in ('torque', 'saturation', 'tracking', 'step'):
        value = float(data.get('rewards', {}).get(key, -1))
        if not math.isfinite(value) or not 0 <= value <= 5:
            raise ValueError(f'{key} weight must be between 0 and 5.')
        rewards[key] = value
    return dict(name=name, policy=data['policy'], steps=steps, learning_rate=rate, rewards=rewards)


def ports_busy():
    if os.name == 'nt':
        import ctypes
        import struct
        size = ctypes.c_ulong(0)
        api = ctypes.WinDLL('iphlpapi').GetTcpTable
        code = api(None, ctypes.byref(size), False)
        if code not in (0, 122):
            return True  # Fail closed if Windows cannot report socket ownership.
        table = ctypes.create_string_buffer(size.value)
        if api(table, ctypes.byref(size), False) != 0:
            return True
        count = struct.unpack_from('<I', table.raw)[0]
        for i in range(count):
            state, _, port, _, _ = struct.unpack_from('<IIIII', table.raw, 4 + i*20)
            if state != 1 and socket.ntohs(port & 65535) in (5582, 5583, 1264, 1265):
                return True
    for port in (5582, 5583, 1264, 1265):
        with socket.socket() as probe:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                return True
    return False


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def tail(path, size=16000):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def status():
    files = [p for pattern in ('*/training_status.json', '*/td3_status.json', '*/sac_status.json') for p in RUNS.glob(pattern)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    busy = ports_busy()
    running = [p for p in files if read_json(p).get('state') in ('training', 'starting')]
    selected = (running[0] if busy and running else files[0]) if files else None
    if PENDING is not None and PENDING.poll() is None:
        pending_files = [p for p in files if read_json(p).get('launcher_pid') == PENDING.pid]
        if pending_files:
            selected = pending_files[0]
    result = dict(busy=busy or (PENDING is not None and PENDING.poll() is None), run=None)
    if selected:
        run = selected.parent
        data = read_json(selected)
        log = tail(run / 'training.log')
        steps = re.findall(r'total_timesteps\s*\|\s*(\d+)', log)
        history = tail(run / 'evaluations/history.jsonl', 30000).splitlines()
        evaluation = next((json.loads(line) for line in reversed(history) if line.startswith('{')), None)
        manifest = read_json(run / 'experiment.json')
        result.update(run=run.name, state=data.get('state'), live=busy and data.get('state') in ('training', 'starting'),
            policy=data.get('algorithm') or manifest.get('algorithm'),
            steps=data.get('total_timesteps') or (int(steps[-1]) if steps else None),
            target=data.get('target_steps') or data.get('requested_steps'), evaluation=evaluation,
            log=log[-10000:], error=data.get('error'), config=read_json(run / 'config.json'))
    return result


class Handler(BaseHTTPRequestHandler):
    def respond(self, body, kind='application/json', code=200):
        payload = json.dumps(body).encode() if kind == 'application/json' else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)

    def trusted_host(self):
        return self.headers.get('Host') in ('127.0.0.1:8765', 'localhost:8765')

    def do_GET(self):
        if not self.trusted_host():
            return self.respond({'error': 'Local host required'}, code=403)
        if self.path == '/':
            return self.respond((HERE / 'training_ui.html').read_text(encoding='utf-8').replace('__TOKEN__', TOKEN), 'text/html')
        if self.path == '/api/status':
            return self.respond(status())
        if self.path.startswith('/api/docs/'):
            key = self.path.removeprefix('/api/docs/')
            if key in DOCS:
                return self.respond({'title': DOCS[key], 'text': (HERE / DOCS[key]).read_text(encoding='utf-8')})
        self.respond({'error': 'Not found'}, code=404)

    def do_POST(self):
        if not self.trusted_host() or self.headers.get('X-Training-Token') != TOKEN:
            return self.respond({'error': 'Reload the local dashboard before making changes.'}, code=403)
        global PENDING
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 10000:
                raise ValueError('Request too large')
            data = json.loads(self.rfile.read(length) or '{}')
            with LOCK:
                if self.path == '/api/start':
                    config = validate_config(data)
                    if ports_busy() or (PENDING is not None and PENDING.poll() is None):
                        raise ValueError('Training or evaluation simulators are already active. Leave the current run running.')
                    if any(read_json(p).get('state') == 'starting' and time.time()-p.stat().st_mtime < 120 for p in RUNS.glob('*/training_status.json')):
                        raise ValueError('A training launch is already starting.')
                    run = RUNS / config['name']
                    run.mkdir(exist_ok=False)
                    (run / 'config.json').write_text(json.dumps(config, indent=2))
                    with (run / 'launcher.log').open('w') as log:
                        PENDING = subprocess.Popen([sys.executable, '-u', str(HERE / 'ui_training_runner.py'), str(run)],
                            cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    (run / 'training_status.json').write_text(json.dumps(dict(state='starting', algorithm=config['policy'],
                        target_steps=config['steps'], launcher_pid=PENDING.pid)))
                    return self.respond({'message': 'Started ' + run.name})
                if self.path == '/api/stop':
                    current = status()
                    if not current.get('live') or data.get('run') != current.get('run'):
                        raise ValueError('The selected run is no longer active. Refresh its status.')
                    (RUNS / current['run'] / 'STOP').touch()
                    return self.respond({'message': 'Stop requested. Waiting for evaluation and model/replay saving.'})
            self.respond({'error': 'Not found'}, code=404)
        except (ValueError, OSError, TypeError) as exc:
            self.respond({'error': str(exc)}, code=400)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    print('Training dashboard: http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
