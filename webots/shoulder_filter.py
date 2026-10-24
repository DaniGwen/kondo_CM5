"""Identical shoulder offset conditioning for Webots training and real deployment.

Input/output: 14 joint offsets in radians. Filter only indices 12 and 13.
State resets at the start of each simulation episode or hardware session.
"""
import numpy as np

SHOULDER_INDICES = (12, 13)
SHOULDER_NEUTRAL_DEADBAND_RAD = 0.008
SHOULDER_CHANGE_THRESHOLD_RAD = 0.008


def filter_shoulders(candidate, previous_sent):
    """Neutral deadband followed by a last-command change threshold.

    Changes smaller than threshold are held. A neutral crossing can therefore
    require multiple control cycles; this avoids rapid tiny command changes.
    """
    result = np.asarray(candidate, dtype=np.float32).copy()
    previous = np.asarray(previous_sent, dtype=np.float32)
    if result.shape != (14,) or previous.shape != (14,):
        raise ValueError('Expected 14 joint offsets for both arguments')
    if not np.all(np.isfinite(result)) or not np.all(np.isfinite(previous)):
        raise ValueError('Non-finite shoulder filter input')
    for i in SHOULDER_INDICES:
        if abs(float(result[i])) < SHOULDER_NEUTRAL_DEADBAND_RAD:
            result[i] = 0.0
        if abs(float(result[i] - previous[i])) < SHOULDER_CHANGE_THRESHOLD_RAD:
            result[i] = previous[i]
    return result
