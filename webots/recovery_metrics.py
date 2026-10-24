"""Recovery outcomes independent of the policy and simulator APIs."""
class RecoveryTracker:
    def __init__(self, tilt_threshold_deg=2.0, stable_steps=10, timeout_steps=125):
        self.threshold = tilt_threshold_deg
        self.required_stable = stable_steps
        self.timeout = timeout_steps
        self.seen = False
        self.stable = 0
        self.peak = 0.0
        self.outcome = None
        self.elapsed_steps = 0

    def update(self, tilt_deg, age, pushing=False, fell=False):
        if self.outcome is not None:
            return self.outcome
        self.peak = max(self.peak, tilt_deg)
        if tilt_deg > self.threshold:
            self.seen = True
            self.stable = 0
        else:
            self.stable = self.stable + 1 if not pushing else 0
        if fell:
            self.outcome = 'failed'
        elif not pushing and self.seen and self.stable >= self.required_stable:
            self.outcome = 'recovered'
        elif age >= self.timeout:
            self.outcome = 'failed' if self.seen else 'no_measurable_disturbance'
        self.elapsed_steps = age
        return self.outcome
