import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Requires training Python environment')
class SACTrainingTests(unittest.TestCase):
    def test_train_save_resume_and_refill(self):
        import gymnasium as gym
        import numpy as np
        import torch
        from stable_baselines3 import SAC
        from train_kondo_balance_sac import parse_args, prepare_run, train

        class FakeBalance(gym.Env):
            observation_space = gym.spaces.Box(-100, 100, (6,), dtype=np.float32)
            action_space = gym.spaces.Box(-.1, .1, (14,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return np.zeros(6, dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                return np.zeros(6, dtype=np.float32), 1.0, False, self.steps >= 8, {}

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with TemporaryDirectory() as temporary:
                argv = ['--run-dir', str(Path(temporary) / 'sac'), '--steps', '32',
                        '--warmup-steps', '8', '--buffer-size', '128',
                        '--batch-size', '8', '--checkpoint-every', '16']
                args = parse_args(argv)
                run = prepare_run(args)
                train(args, FakeBalance(), run)
                model = SAC.load(run / 'kondo_sac.zip')
                self.assertEqual(model.num_timesteps, 32)
                self.assertGreater(model._n_updates, 0)
                self.assertTrue(list((run / 'checkpoints_sac').glob('*.zip')))
                self.assertTrue((run / 'kondo_sac_replay_buffer.pkl').is_file())
                with self.assertRaises(FileExistsError):
                    prepare_run(args)
                args = parse_args(argv + ['--resume'])
                train(args, FakeBalance(), prepare_run(args))
                model = SAC.load(run / 'kondo_sac.zip')
                self.assertEqual(model.num_timesteps, 64)
                updates = model._n_updates
                args = parse_args(argv + ['--resume', '--reset-replay'])
                args.steps = 8
                train(args, FakeBalance(), prepare_run(args))
                model = SAC.load(run / 'kondo_sac.zip')
                self.assertEqual(model.num_timesteps, 72)
                self.assertEqual(model._n_updates, updates)
                self.assertEqual(json.loads((run / 'sac_status.json').read_text())['state'], 'completed')
                # SAC's optional evaluator uses a separate environment and saves
                # its own reference/best actor without loading TD3 weights.
                args = parse_args(['--run-dir', str(Path(temporary) / 'evaluated_sac'), '--steps', '8',
                    '--warmup-steps', '8', '--buffer-size', '128', '--batch-size', '8'])
                fresh = prepare_run(args)
                rows = [dict(force_N=f, angle_deg=a, outcome='recovered', fell=False,
                             peak_tilt_deg=1., outcome_time_s=.5, pre_push_fall=False)
                        for f in (4, 6, 8) for a in (0, 90, 180, 270)]
                with patch.dict('os.environ', {'KONDO_EVAL_PORT': '5583'}), \
                        patch('train_kondo_balance_sac.KondoBalanceEnv', return_value=FakeBalance()), \
                        patch('adaptive_evaluation.evaluate_trials', return_value=rows):
                    train(args, FakeBalance(), fresh)
                self.assertTrue((fresh / 'best_model.zip').is_file())
                self.assertTrue((fresh / 'reference_model.zip').is_file())
                self.assertEqual(len((fresh / 'evaluations/history.jsonl').read_text().splitlines()), 2)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
