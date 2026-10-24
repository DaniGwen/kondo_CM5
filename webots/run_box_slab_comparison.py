"""Evaluate the box collider against existing plane controls without training."""
from pathlib import Path
import subprocess
import sys
from box_slab_world import use_box_slab


def main():
    here = Path(__file__).resolve().parent
    worlds = here.parents[1] / 'webots/kondo_RL/worlds'
    source = worlds / 'kondoKHR3-HV.wbt'
    (worlds / 'kondoKHR3-HV-box-slab.wbt').write_text(use_box_slab(source.read_text(encoding='utf-8')), encoding='utf-8')
    subprocess.run([sys.executable, '-u', str(here / 'investigate_hip_load.py'), '--ablation', 'box_slab'], check=True)
    subprocess.run([sys.executable, '-u', str(here / 'compare_sac_checkpoints.py'), '--box-slab', '--friction', '1',
                    '--output', str(here / 'runs/box_slab_push_comparison_20261023_01')], check=True)


if __name__ == '__main__':
    main()
