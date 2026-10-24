import re
import unittest
from pathlib import Path
from box_slab_world import scale_leg_torque, use_box_slab


class BoxWorldTests(unittest.TestCase):
    def test_only_leg_limits_change(self):
        source = (Path(__file__).resolve().parents[2] / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt').read_text(encoding='utf-8')
        scaled = scale_leg_torque(source, 1.25)
        self.assertEqual(scaled.count('maxTorque 1.716125'), 12)
        self.assertEqual(scaled.count('maxTorque 1.3729'), 10)
        self.assertEqual(re.sub(r'maxTorque\s+[0-9.]+', 'TORQUE', source),
                         re.sub(r'maxTorque\s+[0-9.]+', 'TORQUE', scaled))
        slab = use_box_slab(source)
        self.assertIn('translation 0 0 -0.15', slab)
        self.assertIn('Box { size 1 1 0.1 }', slab)
        self.assertEqual(slab.count('boundingObject USE Wall'), 4)
        self.assertNotIn('RectangleArena {', slab)


if __name__ == '__main__':
    unittest.main()
