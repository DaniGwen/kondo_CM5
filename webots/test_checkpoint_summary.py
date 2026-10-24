import unittest
from evaluate_checkpoints import summarize


class SummaryTests(unittest.TestCase):
    def test_timeouts_are_separate_from_falls_and_baseline_excluded(self):
        rows = [dict(mode='td3', outcome=outcome, fell=str(fell),
                     peak_tilt_deg=str(peak), outcome_time_s=str(duration))
                for outcome, fell, peak, duration in [
                    ('recovered', False, 5, .5), ('failed', False, 4, 2),
                    ('failed', True, 50, .4), ('no_measurable_disturbance', False, 1, 2)]]
        rows.append(dict(mode='baseline', outcome='failed', fell='True'))
        summary = summarize('checkpoint_050000', 50000, rows)
        self.assertEqual(summary['trials'], 4)
        self.assertEqual(summary['falls'], 1)
        self.assertEqual(summary['failed'], 2)
        self.assertEqual(summary['recovered'], 1)
        self.assertEqual(summary['mean_recovery_time_s'], .5)
        self.assertEqual(summary['mean_peak_tilt_deg'], 15)


if __name__ == '__main__':
    unittest.main()
