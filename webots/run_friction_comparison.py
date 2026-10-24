"""Run identical neutral and SAC trials at explicit friction 1, .5 and .25."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    output = here / 'runs/friction_comparison_20261023_01'
    output.mkdir(exist_ok=False)
    status = dict(state='running', completed_settings=0, total_settings=3)
    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2))
    save()
    try:
        for label, friction in [('1', 1), ('0p5', .5), ('0p25', .25)]:
            status['friction'] = friction
            save()
            subprocess.run([sys.executable, '-u', str(here / 'compare_sac_checkpoints.py'),
                '--friction', str(friction), '--output', str(output / ('mu_' + label))], check=True, cwd=here)
            status['completed_settings'] += 1
            save()
        status['state'] = 'completed'
        save()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
