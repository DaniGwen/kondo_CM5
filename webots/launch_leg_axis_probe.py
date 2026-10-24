import os
from pathlib import Path
import subprocess
import argparse
import re
from diagnose_joints import sensor_world
from audit_twin_model import block

parser = argparse.ArgumentParser()
parser.add_argument('--corrected', action='store_true')
args = parser.parse_args()

here = Path(__file__).resolve().parent
root = here.parents[1]
suffix = 'candidate_neutral' if args.corrected else 'original'
output = here / ('runs/leg_joint_verification_20261024_' + suffix)
output.mkdir(exist_ok=False)
world = root / ('webots/kondo_RL/worlds/leg_joint_verification_20261024_' + suffix + '.wbt')
source = (root / 'webots/kondo_RL/worlds/kondoKHR3-HV-box-slab.wbt').read_text(encoding='utf-8')
source = sensor_world(source)
if args.corrected:
    # Use the official neutral endpoint transforms, not saved rotations for old axes.
    upstream = (here / 'runs/twin_free_space_20261023_01/upstream_Khr3hv_R2025a.proto').read_text(encoding='utf-8')
    edits = []
    for match in re.finditer(r'DEF KHR3HV_(\w+) HingeJoint\s*\{', source):
        body_start = source.index('{', match.start()) + 1
        content = block(source, match.start())
        official_match = re.search(r'DEF KHR3HV_' + match.group(1) + r' HingeJoint\s*\{', upstream)
        official = block(upstream, official_match.start())
        endpoint = block(official, official.index('endPoint Solid')).split('children', 1)[0]
        ep_start = source.index('{', body_start + content.index('endPoint Solid')) + 1
        ep_end = source.index('children', ep_start)
        edits.append((ep_start, ep_end, endpoint))
        param_start = source.index('{', body_start + content.index('jointParameters')) + 1
        param_end = source.index('}', param_start)
        params = re.sub(r'\bposition\s+[-+0-9.eE]+', 'position 0', source[param_start:param_end])
        edits.append((param_start, param_end, params))
    for begin, end, replacement in sorted(edits, reverse=True):
        source = source[:begin] + replacement + source[end:]
    for name in ('LeftFemur', 'RightFemur', 'LeftCrus', 'RightCrus', 'RightAnkle'):
        pattern = r'(DEF KHR3HV_' + name + r' HingeJoint\s*\{\s*jointParameters HingeJointParameters\s*\{)([^}]*)(\})'
        def change(match):
            params = re.sub(r'\baxis\s+[-+0-9.eE ]+', '', match.group(2))
            return match.group(1) + '\n axis 0 1 0\n' + params + match.group(3)
        source, count = re.subn(pattern, change, source)
        assert count == 1
source = source.replace('WorldInfo {', 'WorldInfo {\n gravity 0', 1).replace('selfCollision TRUE', 'selfCollision FALSE')
source = source.replace('controller "kondo_tcp_server"', 'controller "leg_axis_probe"')
assert 'controller "leg_axis_probe"' in source
world.write_text(source, encoding='utf-8')
environment = os.environ.copy()
environment['KONDO_AXIS_OUTPUT'] = str(output / 'measured_axes.json')
with (output / 'webots.log').open('w') as log:
    subprocess.run([r'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe', '--batch', '--mode=fast',
        '--no-rendering', '--minimize', '--stdout', '--stderr', '--port=1274', str(world)],
        cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
print('Saved joint sweeps: ' + str(output / 'measured_axes.json'))
