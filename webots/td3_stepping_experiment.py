"""Continue prior TD3 weights with expanded feedback, fresh replay, and step rewards."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def transfer_policy(old, new):
    """Preserve legacy inputs/actions; initialize new observation connections to zero."""
    import torch
    source = old.policy.state_dict()
    target = new.policy.state_dict()
    for name, value in target.items():
        original = source[name]
        if value.shape == original.shape:
            target[name] = original.clone()
        elif value.ndim == 2 and value.shape[0] == original.shape[0]:
            expanded = torch.zeros_like(value)
            expanded[:, :6] = original[:, :6]
            if original.shape[1] == 20:
                expanded[:, -14:] = original[:, -14:]
            elif original.shape[1] != 6:
                raise ValueError('Unexpected first-layer shape: ' + name)
            target[name] = expanded
        else:
            raise ValueError('Unsupported policy shape: ' + name)
    new.policy.load_state_dict(target)


def worker(run):
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from train_kondo_balance import KondoBalanceEnv, BalanceMetricsCallback, StopFileCallback
    from adaptive_evaluation import AdaptiveEvaluationCallback
    from restart_evaluation import restart_evaluation
    manifest = json.loads((run / 'experiment.json').read_text())
    env = KondoBalanceEnv(port=5582)
    eval_env = None
    model = None
    status = dict(state='starting', target_steps=2000000)
    def save():
        (run / 'td3_status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        old = TD3.load(manifest['source_model'], device='cpu') if manifest['source_model'] else None
        noise = NormalActionNoise(mean=np.zeros(14), sigma=np.full(14, .1))
        model = TD3('MlpPolicy', env, policy_kwargs=old.policy_kwargs if old is not None else {},
            learning_rate=1e-4, buffer_size=1000000, learning_starts=5000,
            batch_size=256, action_noise=noise, train_freq=(1, 'step'), gradient_steps=1,
            tensorboard_log=str(run / 'tensorboard_td3'), seed=123, device='cpu', verbose=1)
        if old is not None:
            transfer_policy(old, model)
        model.save(run / 'reference_model.zip')
        eval_env = KondoBalanceEnv(port=5583)
        evaluation = AdaptiveEvaluationCallback(eval_env, run,
            restart=lambda: restart_evaluation(run, 5583))
        evaluation.stage = 1
        callbacks = CallbackList([CheckpointCallback(save_freq=25000, save_path=str(run / 'checkpoints_td3'), name_prefix='td3_step'),
                                  BalanceMetricsCallback(), StopFileCallback(run), evaluation])
        status['state'] = 'training'
        save()
        model.learn(total_timesteps=2000000, callback=callbacks)
        status['state'] = 'stopped' if (run / 'STOP').exists() else 'completed'
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        if model is not None:
            model.save(run / 'kondo_td3.zip')
            model.save_replay_buffer(run / 'replay.pkl')
            status['total_timesteps'] = model.num_timesteps
        env.close()
        if eval_env is not None:
            eval_env.close()
        save()


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    run = here / 'runs/td3_corrected_step_v3_20261024_01'
    run.mkdir(exist_ok=False)
    from correct_leg_model import build_world
    world = build_world()
    settings = dict(KONDO_OBSERVATION_VERSION='3', KONDO_RESET_JOINTS='1', KONDO_WORLD=str(world),
        KONDO_WEBOTS_PORT='5582', KONDO_EVAL_PORT='5583', KONDO_EVAL_WEBOTS_PORT='1265',
        KONDO_CURRICULUM_FILE=str(run / 'curriculum.json'), KONDO_PUSH_MIN_FORCE='.5',
        KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='6', KONDO_PUSH_DURATION_STEPS='10',
        KONDO_NO_PUSH_EPISODES='0', KONDO_PUSH_SETTLE_STEPS='125', KONDO_PUSH_PROBABILITY='.02',
        KONDO_PUSH_COOLDOWN_STEPS='125', KONDO_RANDOM_SEED='12345', PYTHONUNBUFFERED='1')
    environment = os.environ.copy()
    for name in ('KONDO_JOINT_DIAGNOSTICS', 'KONDO_STANCE_DIAGNOSTICS'):
        environment.pop(name, None)
    environment.update(settings)
    (run / 'experiment.json').write_text(json.dumps(dict(algorithm='TD3', source_model=None,
        fresh_replay=True, observation_count=66, additional_steps=2000000, settings=settings,
        transfer='Fresh weights, optimizer and replay: corrected joint kinematics invalidate prior dynamics'), indent=2))
    # Start at 5 N: otherwise the force>4 N recovery-step objective never activates.
    (run / 'curriculum.json').write_text(json.dumps(dict(stage=1, max_force_N=5, gate_streak=0)))
    snapshot = run / 'source_snapshot'
    snapshot.mkdir()
    for name in ('td3_stepping_experiment.py', 'stepping_reward.py', 'balance_reward_v2.py',
                 'train_kondo_balance.py', 'adaptive_evaluation.py', 'adaptive_curriculum.py',
                 'restart_evaluation.py', 'shoulder_filter.py', 'correct_leg_model.py'):
        shutil.copy2(here / name, snapshot / name)
    shutil.copy2(world, snapshot / 'world.wbt')
    shutil.copy2(root / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py', snapshot / 'kondo_tcp_server.py')
    from launch_sac_clean_reset import start_server
    start_server(root, run, environment, 5582, 1264, 'controller.log')
    start_server(root, run, environment, 5583, 1265, 'evaluation_controller.log')
    with (run / 'training.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', str(run)],
                       cwd=here, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
