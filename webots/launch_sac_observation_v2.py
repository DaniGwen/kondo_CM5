"""Start a fresh two-million-step SAC experiment with measured load feedback."""
from pathlib import Path
import subprocess
import sys
def main():
    here = Path(__file__).resolve().parent
    from correct_leg_model import build_world
    world = build_world()
    subprocess.run([sys.executable, '-u', str(here / 'launch_sac_clean_reset.py'),
        '--run-dir', str(here / 'runs/sac_corrected_posture_v2_20261024_01'), '--steps', '2000000',
        '--world', str(world), '--observation-version', '2'], check=True)


if __name__ == '__main__':
    main()
