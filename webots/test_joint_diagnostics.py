from pathlib import Path
import unittest
from diagnose_joints import JOINTS, sensor_world


class DiagnosticWorldTests(unittest.TestCase):
    def test_adds_exactly_one_sensor_per_policy_motor(self):
        path = Path(__file__).resolve().parents[2] / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt'
        original = path.read_text(encoding='utf-8')
        result = sensor_world(original)
        for joint in JOINTS:
            self.assertEqual(result.count('name "diag_' + joint + '"'), 1)
        self.assertEqual(result.count('PositionSensor {'), 14)
        self.assertEqual(result.count('maxTorque'), original.count('maxTorque'))

    def test_missing_or_duplicate_motor_rejected(self):
        with self.assertRaises(ValueError):
            sensor_world('RotationalMotor { name "RightCrus" }')


if __name__ == '__main__':
    unittest.main()
