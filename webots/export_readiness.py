"""Checkpoint-specific export metadata; simulation success is not hardware validation."""
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(run):
    path = run / 'best_model.zip'
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read('data'))
    manifest = {}
    for name in ('experiment.json', 'sac_run.json'):
        if (run / name).exists():
            manifest.update(json.loads((run / name).read_text(encoding='utf-8-sig')))
    shape = saved.get('observation_space', {}).get('_shape', [])
    algorithm = manifest.get('algorithm')
    if algorithm not in ('TD3', 'SAC'):
        raise ValueError('Cannot identify this checkpoint algorithm')
    return dict(run=run.name, policy=algorithm, checkpoint='best_model.zip',
        checkpoint_steps=int(saved.get('num_timesteps', 0)), observation_count=shape[0] if len(shape)==1 else None,
        model_sha256=sha256(path), manifest=manifest)


def assess(run, runner=None):
    data = metadata(run)
    reasons = []
    if data['checkpoint_steps'] < 25000:
        reasons.append('The selected best checkpoint has fewer than 25,000 learning steps; it may still be the initial baseline.')
    if data['observation_count'] != 6:
        reasons.append(f"This policy needs {data['observation_count']} features; the existing robot runner supplies 6. Simulator push/torque/contact features need a hardware-compatible replacement or a new deployable policy.")
    manifest = data['manifest']
    world = manifest.get('settings', {}).get('KONDO_WORLD', '')
    if 'corrected' not in world and 'corrected' not in manifest.get('world_version', ''):
        reasons.append('This run is not recorded against the corrected joint-axis world.')
    runner = runner or Path(__file__).resolve().parent / 'run_RL_balance.py'
    config = {}
    for node in ast.parse(runner.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SERVO_IDS', 'STANDING_ANGLES_DEG'):
                    config[target.id] = ast.literal_eval(node.value)
    ids = config.get('SERVO_IDS', [])
    if len(ids) != 14 or len(set(ids)) != 14 or len(config.get('STANDING_ANGLES_DEG', [])) != 14:
        reasons.append('The hardware runner still needs 14 verified servo IDs and calibrated standing commands.')
    report_path = run / 'deployment_validation.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if report.get('model_sha256') != data['model_sha256']:
        reasons.append('No deployment-validation report is bound to this exact checkpoint hash.')
    for field, label in [('nominal_torque_passed', 'measured real-torque evaluation'),
                         ('sensor_contract_passed', 'hardware observation/units/latency validation'),
                         ('servo_mapping_passed', 'servo direction and travel validation'),
                         ('supported_dry_run_passed', 'supported hardware dry-run validation'),
                         ('recovery_suite_passed', 'hardware-target recovery criteria')]:
        evidence = report.get('evidence', {}).get(field, {})
        relative = evidence.get('file', '')
        path = (run / relative).resolve()
        inside = path.is_relative_to(run.resolve()) and path.is_file()
        valid = inside and evidence.get('sha256') == sha256(path)
        if report.get(field) is not True or not valid:
            reasons.append('Missing verified evidence: ' + label + '.')
    data.pop('manifest')
    data.update(hardware_ready=not reasons, blockers=reasons,
                torque_note='Simulation leg limits are 1.716125 N·m (+25% over the former 1.3729 model value). Real servo capability must be measured; export does not change torque.')
    return data


def candidates(runs):
    paths = sorted(runs.glob('*/best_model.zip'), key=lambda p:p.stat().st_mtime, reverse=True)
    result = []
    for path in paths:
        try:
            result.append(assess(path.parent))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return result
