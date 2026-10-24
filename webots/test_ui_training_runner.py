"""Exercise both UI-selected algorithms without connecting to real simulators."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class RunnerTests(unittest.TestCase):
    def test_both_policies_save_model_and_replay(self):
        import gymnasium as gym
        import numpy as np
        import torch
        import stable_baselines3 as sb3
        from ui_training_runner import worker
        class FakeEnv(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf, (66,), dtype=np.float32)
            action_space = gym.spaces.Box(-.1, .1, (14,), dtype=np.float32)
            def __init__(self, **kwargs):
                self.steps = 0
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return np.zeros(66, np.float32), {}
            def step(self, action):
                self.steps += 1
                return np.zeros(66, np.float32), 1., False, self.steps >= 1000, dict(tilt_rad=0., fell=False)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for name, cls in [('TD3', sb3.TD3), ('SAC', sb3.SAC)]:
                with self.subTest(policy=name), TemporaryDirectory() as temporary:
                    run = Path(temporary)
                    (run / 'config.json').write_text(json.dumps(dict(policy=name, steps=32, learning_rate=.0001)))
                    def small_model(policy, env, **kwargs):
                        kwargs.update(buffer_size=128, learning_starts=4, batch_size=4, policy_kwargs=dict(net_arch=[16,16]), tensorboard_log=None)
                        return cls(policy, env, **kwargs)
                    rows = [dict(force_N=f, angle_deg=a, outcome='recovered', fell=False, peak_tilt_deg=1.) for f in (4,5,6,8) for a in (0,90,180,270)]
                    with patch('train_kondo_balance.KondoBalanceEnv', FakeEnv), patch.object(sb3, name, side_effect=small_model), patch('adaptive_evaluation.evaluate_trials', return_value=rows):
                        worker(run)
                    status = json.loads((run / 'training_status.json').read_text())
                    self.assertEqual(status['state'], 'completed')
                    self.assertEqual(status['total_timesteps'], 32)
                    model = cls.load(run / 'final_model.zip')
                    self.assertGreater(model._n_updates, 0)
                    model.load_replay_buffer(run / 'replay.pkl')
                    self.assertEqual(model.replay_buffer.size(), 32)
                    self.assertTrue((run / 'best_model.zip').is_file())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
