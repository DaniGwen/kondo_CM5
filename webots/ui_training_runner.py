"""Run an immutable UI configuration with independent train/evaluation simulators."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def worker(run):
    import numpy as np
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from train_kondo_balance import KondoBalanceEnv, BalanceMetricsCallback, StopFileCallback
    from adaptive_evaluation import AdaptiveEvaluationCallback
    from restart_evaluation import restart_evaluation
    config = json.loads((run / 'config.json').read_text())
    env = KondoBalanceEnv(port=5582)
    evaluation = None
    model = None
    status = dict(state='starting', algorithm=config['policy'], target_steps=config['steps'], trainer_pid=os.getpid())
    def save():
        (run / 'training_status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        kwargs = dict(learning_rate=config['learning_rate'], buffer_size=1000000, learning_starts=5000,
            batch_size=256, train_freq=(1, 'step'), gradient_steps=1, seed=123, device='cpu',
            tensorboard_log=str(run / 'tensorboard'), verbose=1)
        if config['policy'] == 'TD3':
            kwargs['action_noise'] = NormalActionNoise(mean=np.zeros(14), sigma=np.full(14, .1))
        model = (TD3 if config['policy'] == 'TD3' else SAC)('MlpPolicy', env, **kwargs)
        model.save(run / 'reference_model.zip')
        evaluation = KondoBalanceEnv(port=5583)
        gate = AdaptiveEvaluationCallback(evaluation, run, restart=lambda: restart_evaluation(run, 5583))
        gate.stage = 1
        callbacks = CallbackList([CheckpointCallback(save_freq=25000, save_path=str(run / 'checkpoints'), name_prefix=config['policy'].lower()),
                                  BalanceMetricsCallback(), StopFileCallback(run), gate])
        status['state'] = 'training'
        save()
        model.learn(total_timesteps=config['steps'], callback=callbacks)
        status['state'] = 'stopped' if (run / 'STOP').exists() else 'completed'
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        try:
            if model is not None:
                model.save(run / 'final_model.zip')
                model.save_replay_buffer(run / 'replay.pkl')
                status['total_timesteps'] = model.num_timesteps
        finally:
            env.close()
            if evaluation is not None:
                evaluation.close()
            save()


def main(run):
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    config = json.loads((run / 'config.json').read_text())
    from correct_leg_model import build_world
    # Snapshot the world per run so later edits cannot change evaluation restarts.
    world = build_world()
    local_world = world.with_name(run.name + '.wbt')
    shutil.copy2(world, local_world)
    settings = dict(KONDO_OBSERVATION_VERSION='3', KONDO_RESET_JOINTS='1', KONDO_WORLD=str(local_world),
        KONDO_WEBOTS_PORT='5582', KONDO_EVAL_PORT='5583', KONDO_EVAL_WEBOTS_PORT='1265',
        KONDO_CURRICULUM_FILE=str(run / 'curriculum.json'), KONDO_PUSH_MIN_FORCE='.5',
        KONDO_PUSH_MAX_FORCE_INITIAL='4', KONDO_PUSH_MAX_FORCE_FINAL='6', KONDO_PUSH_DURATION_STEPS='10',
        KONDO_NO_PUSH_EPISODES='0', KONDO_PUSH_SETTLE_STEPS='125', KONDO_PUSH_PROBABILITY='.02',
        KONDO_PUSH_COOLDOWN_STEPS='125', KONDO_RANDOM_SEED='12345', PYTHONUNBUFFERED='1')
    settings.update({f'KONDO_REWARD_{key.upper()}': str(value) for key, value in config['rewards'].items()})
    environment = os.environ.copy()
    for key in ('KONDO_JOINT_DIAGNOSTICS', 'KONDO_STANCE_DIAGNOSTICS', 'KONDO_RESET_ALL_TARGETS'):
        environment.pop(key, None)
    environment.update(settings)
    (run / 'experiment.json').write_text(json.dumps(dict(algorithm=config['policy'], fresh_replay=True,
        settings=settings, config=config, world_version='corrected-20261024', observation_count=66), indent=2))
    (run / 'curriculum.json').write_text(json.dumps(dict(stage=1, max_force_N=5, gate_streak=0)))
    snapshot = run / 'source_snapshot'
    snapshot.mkdir()
    for name in ('ui_training_runner.py', 'train_kondo_balance.py', 'balance_reward_v2.py', 'stepping_reward.py',
                 'adaptive_evaluation.py', 'adaptive_curriculum.py', 'restart_evaluation.py', 'shoulder_filter.py'):
        shutil.copy2(here / name, snapshot / name)
    shutil.copy2(local_world, snapshot / 'world.wbt')
    shutil.copy2(root / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py', snapshot / 'kondo_tcp_server.py')
    from launch_sac_clean_reset import start_server
    start_server(root, run, environment, 5582, 1264, 'controller.log')
    start_server(root, run, environment, 5583, 1265, 'evaluation_controller.log')
    with (run / 'training.log').open('w') as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', str(run)], env=environment,
                       cwd=here, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    run = Path(sys.argv[-1]).resolve()
    if sys.argv[1] == '--worker':
        worker(run)
    else:
        try:
            main(run)
        except Exception as exc:
            (run / 'training_status.json').write_text(json.dumps(dict(state='failed', error=str(exc))))
            raise
