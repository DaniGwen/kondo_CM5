"""Real-simulator neutral hold and short SAC update check before a long run."""
import json
import os
import argparse
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=('2', '3'), default='2')
    parser.add_argument('--tag', default='')
    parser.add_argument('--world', type=Path)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    v3 = args.version == '3'
    count = 66 if v3 else 51
    run = here / ('runs/stepping_v3_validation_20261024_01' if v3 else 'runs/observation_v2_validation_20261023_01')
    if args.tag:
        run = run.with_name(run.name + args.tag)
    run.mkdir(exist_ok=False)
    from diagnose_joints import sensor_world
    worlds = here.parents[1] / 'webots/kondo_RL/worlds'
    world = worlds / ('kondoKHR3-HV-box-slab-torque125-stepping-v3.wbt' if v3 else 'kondoKHR3-HV-box-slab-torque125-observation-v2.wbt')
    if args.world:
        world = args.world.resolve()
    else:
        world.write_text(sensor_world((worlds / 'kondoKHR3-HV-box-slab-torque125.wbt').read_text(encoding='utf-8')), encoding='utf-8')
    os.environ.update(KONDO_OBSERVATION_VERSION=args.version, KONDO_RESET_JOINTS='1', KONDO_WORLD=str(world),
                     KONDO_EVAL_WEBOTS_PORT='1272', KONDO_WEBOTS_PORT='5590')
    for name in ('KONDO_CURRICULUM_FILE', 'KONDO_EVAL_PORT', 'KONDO_JOINT_DIAGNOSTICS', 'KONDO_STANCE_DIAGNOSTICS'):
        os.environ.pop(name, None)
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from restart_evaluation import restart_evaluation
    torch.set_num_threads(1)
    env = restart_evaluation(run, 5590)
    try:
        obs, _ = env.reset(options={'push_trial': (0, 0, 10, 2000)})
        assert obs.shape == (count,)
        loads, rewards = [], []
        for step in range(625):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(14))
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            assert not terminated and not truncated
            if step >= 500:
                loads.append(float(np.mean(np.abs(obs[[24, 30]]))))
                rewards.append(reward)
        assert np.mean(loads) < .3
        contacts = list(map(float, obs[59:61])) if v3 else []
        if v3:
            assert contacts == [1., 1.], 'Neutral feet must register floor contact'
            for angle, sign in ((0, 1), (180, -1)):
                obs, _ = env.reset(options={'push_trial': (5, angle, 10, 125)})
                for _ in range(125):
                    obs, _, done, _, _ = env.step(np.zeros(14))
                    assert not done
                assert sign * obs[61] > 4.9 and abs(obs[62]) < .1
        if v3:
            from stable_baselines3 import TD3
            from td3_stepping_experiment import transfer_policy
            old = TD3.load(here / 'runs/adaptive_mixed160_20261021_01/best_model.zip', device='cpu')
            model = TD3('MlpPolicy', env, policy_kwargs=old.policy_kwargs, learning_starts=64,
                        buffer_size=512, batch_size=32, device='cpu', seed=123)
            if not args.fresh:
                transfer_policy(old, model)
            new_obs = torch.randn(3, count)
            action = torch.randn(3, 14)
            with torch.no_grad():
                if not args.fresh:
                    assert torch.allclose(model.actor(new_obs), old.actor(new_obs[:, :6]), atol=1e-6)
                    for new_q, old_q in zip(model.critic(new_obs, action), old.critic(new_obs[:, :6], action)):
                        assert torch.allclose(new_q, old_q, atol=1e-6)
        else:
            model = SAC('MlpPolicy', env, learning_starts=64, buffer_size=512, batch_size=32, device='cpu', seed=123)
        model.learn(total_timesteps=256)
        assert model._n_updates > 0
        model.save(run / 'smoke_model.zip')
        summary = dict(state='passed', observation_count=count, neutral_hold_steps=625,
                       mean_hip_load_fraction=float(np.mean(loads)), mean_neutral_reward=float(np.mean(rewards)),
                       smoke_steps=model.num_timesteps, gradient_updates=model._n_updates, neutral_foot_contacts=contacts)
        (run / 'validation.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    finally:
        env.close()


if __name__ == '__main__':
    main()
