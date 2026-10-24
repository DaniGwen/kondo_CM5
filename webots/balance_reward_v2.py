"""Reward for measured posture, tracking, and available actuator reserve."""
import numpy as np
import os

TORQUE_WEIGHT = float(os.getenv('KONDO_REWARD_TORQUE', '.75'))
SATURATION_WEIGHT = float(os.getenv('KONDO_REWARD_SATURATION', '.75'))
TRACKING_WEIGHT = float(os.getenv('KONDO_REWARD_TRACKING', '.25'))


def load_penalties(obs):
    torque = np.clip(np.abs(obs[23:35]), 0, 2)
    error = np.clip(obs[37:49], -2, 2)
    return dict(torque=TORQUE_WEIGHT * float(np.mean(torque ** 2)),
                saturation=SATURATION_WEIGHT * float(np.mean(np.clip((torque - .8) / .2, 0, 1) ** 2)),
                tracking=TRACKING_WEIGHT * float(np.mean(error ** 2)))


def revised_reward(original_reward, obs):
    return original_reward - sum(load_penalties(obs).values())
