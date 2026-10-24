import unittest
from recovery_metrics import RecoveryTracker


class RecoveryTests(unittest.TestCase):
    def test_unmeasurable_is_not_failure(self):
        tracker = RecoveryTracker(timeout_steps=5)
        for age in range(1, 6):
            tracker.update(1.5, age)
        self.assertEqual(tracker.outcome, 'no_measurable_disturbance')

    def test_recovery_requires_excursion_and_stability_after_push(self):
        tracker = RecoveryTracker(stable_steps=2)
        tracker.update(4, 1, pushing=True)
        tracker.update(1, 2, pushing=True)
        tracker.update(1, 3)
        self.assertIsNone(tracker.outcome)
        self.assertEqual(tracker.update(1, 4), 'recovered')

    def test_timeout_and_fall_are_failures(self):
        tracker = RecoveryTracker(timeout_steps=3)
        for age in range(1, 4):
            tracker.update(4, age)
        self.assertEqual(tracker.outcome, 'failed')
        self.assertEqual(RecoveryTracker().update(1, 1, fell=True), 'failed')


if __name__ == '__main__':
    unittest.main()
