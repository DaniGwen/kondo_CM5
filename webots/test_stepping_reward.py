import unittest
from stepping_reward import RecoveryStepReward


class StepTests(unittest.TestCase):
    def trial(self, force, direction, landing_direction, lift=True):
        reward = RecoveryStepReward()
        obs = [0.] * 66
        obs[61] = direction * force
        obs[63] = .2
        if lift:
            obs[57] = .01
        self.assertEqual(reward.update(obs, .05, False), 0)
        obs[57], obs[59], obs[63], obs[64] = 0, 1, .5, landing_direction * .02
        result = reward.update(obs, .05, False)
        self.assertEqual(reward.update(obs, .05, False), 0)
        return result

    def test_forward_and_backward_landing(self):
        self.assertEqual(self.trial(5, 1, 1), .75)
        self.assertEqual(self.trial(5, -1, -1), .75)

    def test_no_bonus_for_weak_wrong_direction_or_sliding(self):
        self.assertEqual(self.trial(4, 1, 1), 0)
        self.assertEqual(self.trial(5, 1, -1), 0)
        self.assertEqual(self.trial(5, 1, 1, lift=False), 0)


if __name__ == '__main__':
    unittest.main()
