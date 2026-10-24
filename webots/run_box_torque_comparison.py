"""Test a 25% leg torque increase on the static box slab."""
from pathlib import Path
import subprocess
import sys
from box_slab_world import use_box_slab, scale_leg_torque


def main():
    here = Path(__file__).resolve().parent
    worlds = here.parents[1] / 'webots/kondo_RL/worlds'
    source = (worlds / 'kondoKHR3-HV.wbt').read_text(encoding='utf-8')
    world = worlds / 'kondoKHR3-HV-box-slab-torque125.wbt'
    world.write_text(scale_leg_torque(use_box_slab(source), 1.25), encoding='utf-8')
    subprocess.run([sys.executable, '-u', str(here / 'investigate_hip_load.py'),
                    '--ablation', 'box_slab', '--torque-scale', '1.25'], check=True)
    subprocess.run([sys.executable, '-u', str(here / 'compare_sac_checkpoints.py'),
                    '--box-slab', '--friction', '1', '--torque-scale', '1.25',
                    '--output', str(here / 'runs/box_slab_torque125_20261023_01')], check=True)


if __name__ == '__main__':
    main()
