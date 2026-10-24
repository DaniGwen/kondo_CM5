import ast
import math
import os
from pathlib import Path
import socket
import types
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from balance_reward_v2 import load_penalties, revised_reward


class ObservationRewardTests(unittest.TestCase):
    def test_saturated_hold_blocks_promotion(self):
        from adaptive_evaluation import AdaptiveEvaluationCallback
        for load, expected in ((1., False), (.2, True)):
            with self.subTest(load=load), TemporaryDirectory() as temporary:
                obs = np.zeros(51, dtype=np.float32)
                obs[[24, 30]] = load
                env = types.SimpleNamespace(observation_space=types.SimpleNamespace(shape=(51,)),
                    reset=lambda **kwargs: (obs, {}),
                    step=lambda action: (obs, 1., False, False, {'fell': False}))
                saves = []
                model = types.SimpleNamespace(policy=types.SimpleNamespace(training=False, set_training_mode=lambda value: None),
                    predict=lambda obs, **kwargs: (np.zeros(14), None), save=lambda path: saves.append(path))
                callback = AdaptiveEvaluationCallback(env, Path(temporary))
                callback.model = model
                rows = [dict(force_N=force, angle_deg=angle, outcome='recovered', fell=False, peak_tilt_deg=1.)
                        for force in (4, 6, 8) for angle in (0, 90, 180, 270)]
                callback.reference = rows
                callback.best_rank = (1, 1, 10.)
                callback.start_steps = 0
                callback.num_timesteps = 100
                with patch('adaptive_evaluation.evaluate_trials', return_value=rows):
                    callback.evaluate()
                self.assertEqual(bool(saves), expected)
                self.assertEqual(callback.gate_streak, int(expected))

    def test_loaded_stance_is_worse_than_neutral(self):
        neutral = np.zeros(51)
        neutral[23:35] = .1
        loaded = neutral.copy()
        loaded[[24, 30]] = 1
        loaded[[38, 44]] = .5
        self.assertLess(revised_reward(3, loaded), revised_reward(3, neutral) - .25)
        self.assertGreater(load_penalties(loaded)['saturation'], 0)
        loaded[23:35] *= -1
        self.assertAlmostEqual(revised_reward(3, loaded), revised_reward(3, np.abs(loaded)))

    def test_server_client_v2_packet_and_reset_ack(self):
        from train_kondo_balance import KondoBalanceEnv, OBSERVATION_COUNT
        self.assertEqual(OBSERVATION_COUNT, 51)
        path = Path(__file__).resolve().parents[2] / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('read_policy_observation', 'read_telemetry', 'send_state')]
        motor = types.SimpleNamespace(getTorqueFeedback=lambda: .5, getMaxTorque=lambda: 2, getTargetPosition=lambda: .03)
        namespace = dict(STEPPING=False, EXTENDED_OBSERVATION=True, STATE_COUNT=53, math=math, socket=socket, struct=__import__('struct'),
            accel=types.SimpleNamespace(getValues=lambda: [1, 2, 3]), gyro=types.SimpleNamespace(getValues=lambda: [4, 5, 6]),
            kondo_node=types.SimpleNamespace(getOrientation=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1], getPosition=lambda: [0, 0, .24]),
            policy_joint_sensors=[types.SimpleNamespace(getValue=lambda: .02)] * 14,
            motors=[motor]*14, ACTION_LIMITS=[.1]*12+[.2]*2)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), namespace)
        sender, receiver = socket.socketpair()
        with sender, receiver:
            namespace['send_state'](sender, evaluation_ack=True)
            env = KondoBalanceEnv.__new__(KondoBalanceEnv)
            env.sock = receiver
            obs, tilt, height = env._receive_state()
        self.assertEqual(obs.shape, (51,))
        self.assertEqual(obs[5], -999)
        np.testing.assert_allclose(obs[6:9], [0, 0, 1])
        np.testing.assert_allclose(obs[9:21], .2)
        np.testing.assert_allclose(obs[23:37], .25)
        np.testing.assert_allclose(obs[37:49], .1)
        self.assertEqual(tilt, 0)
        self.assertAlmostEqual(height, .24)


if __name__ == '__main__':
    unittest.main()
