"""Create a parity-tested ONNX package. Never sends robot commands."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
if (HERE / '.export_tools').is_dir():
    sys.path.insert(0, str(HERE / '.export_tools'))


def create_package(run, output_root, hardware=False):
    import numpy as np
    import torch
    from torch import nn
    from stable_baselines3 import TD3, SAC
    import onnx
    import onnxruntime as ort
    from export_readiness import assess, sha256
    torch.set_num_threads(1)
    assessment = assess(run)
    if hardware and not assessment['hardware_ready']:
        raise ValueError('Real-robot export blocked: ' + ' '.join(assessment['blockers']))
    name = 'export_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3)
    output = output_root / name
    output.mkdir(parents=True, exist_ok=False)
    # Copy first; never load a checkpoint that may be overwritten during an evaluation.
    shutil.copy2(run / 'best_model.zip', output / 'checkpoint.zip')
    if sha256(output / 'checkpoint.zip') != assessment['model_sha256']:
        raise ValueError('Checkpoint changed during export. Retry after evaluation finishes.')
    cls = TD3 if assessment['policy'] == 'TD3' else SAC
    model = cls.load(output / 'checkpoint.zip', device='cpu')
    shape = model.observation_space.shape
    if len(shape) != 1 or model.action_space.shape != (14,):
        raise ValueError('Only vector observations and fourteen joint actions are supported')
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = model.policy.actor
            self.register_buffer('scale', torch.as_tensor((model.action_space.high-model.action_space.low)/2))
            self.register_buffer('bias', torch.as_tensor((model.action_space.high+model.action_space.low)/2))
        def forward(self, observation):
            action = self.actor(observation, deterministic=True) if assessment['policy'] == 'SAC' else self.actor(observation)
            return action*self.scale+self.bias
    wrapper = Actor().eval()
    onnx_path = output / 'policy.onnx'
    torch.onnx.export(wrapper, torch.zeros((1, shape[0]), dtype=torch.float32), str(onnx_path),
        input_names=['observation'], output_names=['joint_actions'], opset_version=17,
        dynamic_axes={'observation':{0:'batch'}, 'joint_actions':{0:'batch'}}, dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    error = 0.
    for batch in (1, 8):
        observations = np.random.default_rng(17+batch).normal(size=(batch,shape[0])).astype(np.float32)
        expected = model.predict(observations, deterministic=True)[0]
        actual = runtime.run(['joint_actions'], {'observation':observations})[0]
        if not np.all(np.isfinite(actual)) or actual.shape != (batch,14):
            raise ValueError('Invalid ONNX output')
        error = max(error, float(np.max(np.abs(expected-actual))))
    if error > 1e-4:
        raise ValueError('ONNX parity check failed: ' + str(error))
    contract = dict(input='observation', shape=['batch', shape[0]], dtype='float32', output='joint_actions',
        output_units='joint offsets in radians, scaled to training action bounds',
        action_low=model.action_space.low.tolist(), action_high=model.action_space.high.tolist(),
        filtering=dict(alpha=.25, delta_limits=[.03]*12+[.04]*2, shoulder_deadband=.008, shoulder_hysteresis=.008),
        note='ONNX produces raw deterministic actions. The matching observation pipeline, filtering, safety limits and servo conversion must run separately.')
    assessment.update(onnx_parity_max_abs_error=error, onnx_sha256=sha256(onnx_path),
        export_type='hardware_candidate' if hardware else 'research_only', hardware_control_enabled=False)
    (output / 'manifest.json').write_text(json.dumps(assessment, indent=2))
    (output / 'interface.json').write_text(json.dumps(contract, indent=2))
    for doc in ('ROBOT_EXPORT.md', 'OBSERVATION_V2.md', 'TD3_STEPPING_V3.md', 'CORRECTED_MODEL.md'):
        shutil.copy2(HERE / doc, output / doc)
    for source in ('experiment.json', 'config.json', 'deployment_validation.json'):
        if (run / source).exists():
            shutil.copy2(run / source, output / source)
    warning = 'RESEARCH EXPORT — NOT APPROVED FOR REAL-ROBOT CONTROL' if not hardware else 'HARDWARE CANDIDATE — CONTROL REMAINS DISABLED'
    (output / 'README.txt').write_text(warning + '\n\n' + '\n'.join(assessment['blockers']) +
        '\n\nNo robot commands are sent. No physical torque setting is changed. Read ROBOT_EXPORT.md and interface.json before use.\n')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output.iterdir():
            bundle.write(path, arcname=path.name)
    return dict(filename=archive.name, readiness=assessment, path=str(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    args = parser.parse_args()
    print(json.dumps(create_package(args.run.resolve(), args.output_root.resolve(), args.hardware)))
