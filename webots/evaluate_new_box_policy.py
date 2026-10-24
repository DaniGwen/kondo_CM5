"""Matched previous/new SAC/neutral evaluation plus no-push torque checks."""
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    model = here / 'runs/sac_box_torque125_20261023_01/best_model.zip'
    subprocess.run([sys.executable, '-u', str(here / 'compare_sac_checkpoints.py'),
        '--box-slab', '--torque-scale', '1.25', '--friction', '1', '--forces', '4', '5', '6', '8',
        '--new-best', str(model), '--output', str(here / 'runs/sac_box_torque125_evaluation_20261023_01')], check=True)
    subprocess.run([sys.executable, '-u', str(here / 'investigate_hip_load.py'),
        '--ablation', 'box_slab', '--torque-scale', '1.25', '--best-model', str(model), '--tag', '_new_policy'], check=True)


if __name__ == '__main__':
    main()
