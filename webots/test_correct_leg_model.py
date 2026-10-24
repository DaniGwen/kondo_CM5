from pathlib import Path
import unittest
from correct_leg_model import corrected_legs
from audit_twin_model import joints
from report_leg_axes import report


class CorrectLegTests(unittest.TestCase):
    def test_axes_and_limits(self):
        here = Path(__file__).resolve().parent
        worlds = here.parents[1] / 'webots/kondo_RL/worlds'
        before = (worlds / 'kondoKHR3-HV-box-slab-torque125.wbt').read_text(encoding='utf-8')
        official = (here / 'runs/twin_free_space_20261023_01/upstream_Khr3hv_R2025a.proto').read_text(encoding='utf-8')
        corrected = corrected_legs(before, official)
        original, actual = joints(before), joints(corrected)
        changed = {'LeftFemur', 'RightFemur', 'LeftCrus', 'RightCrus', 'RightAnkle'}
        self.assertEqual(len(actual), 22)
        for name, joint in actual.items():
            self.assertEqual(joint['saved_position_rad'], 0)
            self.assertEqual(joint['motor'], original[name]['motor'])
            self.assertEqual(joint['anchor'], original[name]['anchor'])
            self.assertEqual(joint['axis'], [0, 1, 0] if name in changed else original[name]['axis'])
        self.assertIn('selfCollision TRUE', corrected)
        self.assertNotIn('gravity 0', corrected)


if __name__ == '__main__':
    unittest.main()
