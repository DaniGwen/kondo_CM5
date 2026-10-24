"""Reward one supported, directed recovery landing per foot after a strong push."""
import math
import os

STEP_BONUS = float(os.getenv('KONDO_REWARD_STEP', '.75'))


class RecoveryStepReward:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_age = 100.
        self.lifted = [False, False]
        self.paid = [False, False]

    def update(self, obs, tilt, fell):
        self.last_landings = 0
        force_x, force_y, age = map(float, obs[61:64])
        if age < self.last_age:
            self.lifted = [False, False]
            self.paid = [False, False]
        self.last_age = age
        magnitude = math.hypot(force_x, force_y)
        if magnitude <= 4 or abs(force_x) < .7 * magnitude or fell or abs(tilt) > .35:
            return 0.
        sign = 1 if force_x > 0 else -1
        reward = 0.
        for i in range(2):
            if obs[57+i] >= .005 and obs[59+i] < .5:
                self.lifted[i] = True
            if (self.lifted[i] and not self.paid[i] and obs[59+i] >= .5 and
                    obs[57+i] <= .003 and sign * obs[64+i] >= .01):
                reward += STEP_BONUS
                self.last_landings += 1
                self.paid[i] = True
        return reward
