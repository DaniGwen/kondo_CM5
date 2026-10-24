import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import json
import zipfile
from unittest.mock import patch
from adaptive_curriculum import can_advance, preserves_reference, rank


def trials(fail=False):
    return [dict(force_N=f, angle_deg=a, outcome='failed' if fail and a == 180 else 'recovered',
                 fell=fail and a == 180, peak_tilt_deg=5.0, outcome_time_s=.5)
            for f in (4, 6, 8) for a in (0, 90, 180, 270)]


class CurriculumTests(unittest.TestCase):
    def test_resume_keeps_original_baseline_and_best(self):
        if importlib.util.find_spec('stable_baselines3') is None:
            self.skipTest('SB3 not installed in this test runtime')
        from adaptive_evaluation import AdaptiveEvaluationCallback
        with TemporaryDirectory() as directory:
            policy = SimpleNamespace(training=True, set_training_mode=lambda mode: None)
            model = SimpleNamespace(policy=policy, num_timesteps=100, save=lambda path: None)
            callback = AdaptiveEvaluationCallback(SimpleNamespace(), directory)
            callback.model = model
            callback.num_timesteps = 100
            with patch('adaptive_evaluation.evaluate_trials', return_value=trials()):
                callback._on_training_start()
            with zipfile.ZipFile(Path(directory) / 'reference_model.zip', 'w') as archive:
                archive.writestr('data', json.dumps({'num_timesteps': 100}))
            saves = []
            resumed = AdaptiveEvaluationCallback(SimpleNamespace(), directory)
            resumed.model = SimpleNamespace(policy=policy, num_timesteps=25100, save=saves.append)
            resumed.num_timesteps = 25100
            with patch('adaptive_evaluation.evaluate_trials', return_value=trials(fail=True)):
                resumed._on_training_start()
            self.assertEqual(resumed.start_steps, 100)
            self.assertEqual(resumed.best_rank, (0, 0, 5.0))
            self.assertTrue(all(r['outcome'] == 'recovered' for r in resumed.reference))
            self.assertEqual(saves, [])

    def test_disconnected_evaluator_is_restarted_and_suite_retried(self):
        if importlib.util.find_spec('stable_baselines3') is None:
            self.skipTest('SB3 not installed in this test runtime')
        from adaptive_evaluation import AdaptiveEvaluationCallback
        from unittest.mock import Mock
        with TemporaryDirectory() as directory:
            env = Mock()
            replacement = Mock()
            restart = Mock(return_value=replacement)
            callback = AdaptiveEvaluationCallback(env, directory, restart=restart)
            callback.model = SimpleNamespace(policy=SimpleNamespace(training=True,
                set_training_mode=lambda mode: None), num_timesteps=100, save=lambda path: None)
            callback.num_timesteps = 100
            with patch('adaptive_evaluation.evaluate_trials', side_effect=[ConnectionResetError(), trials()]) as evaluate:
                callback._on_training_start()
            restart.assert_called_once()
            env.close.assert_called_once()
            self.assertIs(evaluate.call_args.args[1], replacement)

    def test_regression_blocks_advancement(self):
        reference = trials()
        candidate = trials(fail=True)
        gate = [r for r in candidate if r['force_N'] == 4]
        self.assertFalse(preserves_reference(reference, candidate))
        self.assertFalse(can_advance(reference, candidate, gate))

    def test_gate_requires_measurable_recovery(self):
        rows = trials()
        gate = [dict(r, outcome='no_measurable_disturbance') for r in rows[:4]]
        self.assertFalse(can_advance(rows, rows, gate))
        gate[0]['outcome'] = 'recovered'
        self.assertTrue(can_advance(rows, rows, gate))

    def test_failures_rank_before_peak_tilt(self):
        self.assertLess(rank(trials()), rank(trials(fail=True)))

    def test_callback_keeps_original_and_advances_after_two_passes(self):
        if importlib.util.find_spec('stable_baselines3') is None:
            self.skipTest('SB3 not installed in this test runtime')
        from adaptive_evaluation import AdaptiveEvaluationCallback
        with TemporaryDirectory() as directory:
            env = SimpleNamespace(close=lambda: None)
            saved = []
            policy = SimpleNamespace(training=True, set_training_mode=lambda mode: None)
            callback = AdaptiveEvaluationCallback(env, directory)
            callback.model = SimpleNamespace(policy=policy, num_timesteps=100, save=saved.append)
            callback.num_timesteps = 100
            with patch('adaptive_evaluation.evaluate_trials', return_value=trials()):
                callback._on_training_start()
                callback.num_timesteps += 25000
                callback._on_step()
                self.assertEqual(callback.stage, 0)
                callback.num_timesteps += 25000
                callback._on_step()
                self.assertEqual(callback.stage, 1)
            self.assertEqual(len(saved), 1)  # A tie cannot replace the original best.
            with patch('adaptive_evaluation.evaluate_trials', return_value=trials(fail=True)):
                callback.num_timesteps += 25000
                callback._on_step()
                self.assertEqual(callback.gate_streak, 0)
            self.assertTrue((Path(directory) / 'evaluations/history.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
