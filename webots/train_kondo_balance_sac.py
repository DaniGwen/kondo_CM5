"""Train a new SAC balance policy using the existing Webots environment.

Start the Webots TCP controller first. This script does not start a simulator.
SAC models are independent of TD3; --resume accepts only this script's run folders.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from train_kondo_balance import KondoBalanceEnv, BalanceMetricsCallback, StopFileCallback
from train_kondo_balance import OBSERVATION_COUNT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent /
                        'runs' / ('sac_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reset-replay', action='store_true')
    parser.add_argument('--host', default=os.getenv('KONDO_WEBOTS_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('KONDO_WEBOTS_PORT', '5555')))
    parser.add_argument('--steps', type=int, default=500_000, help='Additional steps for this invocation')
    parser.add_argument('--warmup-steps', type=int, default=5000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--checkpoint-every', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if (min(args.steps, args.warmup_steps, args.buffer_size, args.batch_size, args.checkpoint_every) < 1 or
            args.buffer_size < max(args.warmup_steps, args.batch_size) or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not 1 <= args.port <= 65535):
        parser.error('Invalid training settings; buffer must hold warmup and batch sizes')
    if args.reset_replay and not args.resume:
        parser.error('--reset-replay requires --resume')
    return args


def prepare_run(args):
    run = args.run_dir.resolve()
    if args.resume:
        manifest = json.loads((run / 'sac_run.json').read_text(encoding='utf-8'))
        if manifest.get('algorithm') != 'SAC' or not (run / 'kondo_sac.zip').is_file():
            raise ValueError('Resume requires an existing SAC run, not a TD3 checkpoint')
        if manifest.get('observation_count', 6) != OBSERVATION_COUNT:
            raise ValueError('Observation version changed: start a fresh run and replay')
    else:
        run.mkdir(parents=True, exist_ok=False)
        manifest = dict(algorithm='SAC', created_at=datetime.now().astimezone().isoformat(),
                        observation_count=OBSERVATION_COUNT, action_count=14,
                        environment='KondoBalanceEnv; reward v2' if OBSERVATION_COUNT == 51 else 'KondoBalanceEnv; reward v1',
                        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        (run / 'sac_run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if (run / 'STOP').exists():
        raise ValueError('Remove the STOP file before resuming training')
    return run


def train(args, env, run):
    model_path = run / 'kondo_sac.zip'
    replay_path = run / 'kondo_sac_replay_buffer.pkl'
    status = dict(state='starting', algorithm='SAC', requested_steps=args.steps)
    def save_status():
        (run / 'sac_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    model = None
    can_save = False
    save_status()
    try:
        if args.resume:
            model = SAC.load(str(model_path), env=env, device=args.device,
                             learning_rate=args.learning_rate, seed=args.seed,
                             tensorboard_log=str(run / 'tensorboard_sac'))
            if replay_path.exists() and not args.reset_replay:
                model.load_replay_buffer(str(replay_path))
                print(f'Loaded SAC replay: {model.replay_buffer.size()} transitions', flush=True)
        else:
            model = SAC('MlpPolicy', env, learning_rate=args.learning_rate,
                        buffer_size=args.buffer_size, learning_starts=args.warmup_steps,
                        batch_size=args.batch_size, tau=.005, gamma=.99,
                        train_freq=(1, 'step'), gradient_steps=1,
                        ent_coef='auto', target_entropy='auto',
                        tensorboard_log=str(run / 'tensorboard_sac'),
                        seed=args.seed, device=args.device, verbose=1)
        # SAC explores through its stochastic policy; no TD3 action-noise object.
        missing = max(0, args.warmup_steps - model.replay_buffer.size())
        if missing:
            model.learning_starts = max(model.learning_starts, model.num_timesteps + missing)
            print(f'Collecting {missing} SAC warm-up steps before updates', flush=True)
        starting_steps = model.num_timesteps
        can_save = True
        status.update(state='training', starting_steps=starting_steps)
        save_status()
        callbacks = CallbackList([
            CheckpointCallback(save_freq=args.checkpoint_every,
                               save_path=str(run / 'checkpoints_sac'), name_prefix='kondo_sac'),
            BalanceMetricsCallback(), StopFileCallback(run)])
        if os.getenv('KONDO_EVAL_PORT'):
            from adaptive_evaluation import AdaptiveEvaluationCallback
            from restart_evaluation import restart_evaluation
            reference = run / 'reference_model.zip'
            if not reference.exists():
                model.save(str(reference))
            evaluation_port = int(os.environ['KONDO_EVAL_PORT'])
            evaluation_env = KondoBalanceEnv(host=args.host, port=evaluation_port)
            callbacks.callbacks.append(AdaptiveEvaluationCallback(
                evaluation_env, run, restart=lambda: restart_evaluation(run, evaluation_port)))
        model.learn(total_timesteps=args.steps, reset_num_timesteps=False, callback=callbacks)
        status['state'] = 'stopped' if (run / 'STOP').exists() else 'completed'
    except KeyboardInterrupt:
        status['state'] = 'interrupted'
        print('Interrupted; saving SAC model and replay', flush=True)
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        try:
            if model is not None and can_save:
                model.save(str(model_path))
                model.save_replay_buffer(str(replay_path))
                status['total_timesteps'] = model.num_timesteps
                print(f'Saved SAC model and replay in {run}', flush=True)
        finally:
            env.close()
            save_status()


def main():
    args = parse_args()
    run = prepare_run(args)
    print(f'SAC run: {run}; Webots {args.host}:{args.port}', flush=True)
    print('Training from scratch' if not args.resume else 'Resuming SAC', flush=True)
    env = KondoBalanceEnv(host=args.host, port=args.port)
    train(args, env, run)


if __name__ == '__main__':
    main()
