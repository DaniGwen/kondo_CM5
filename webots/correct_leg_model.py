"""Build the verified neutral-pose, corrected-axis training world."""
from pathlib import Path
import re
from audit_twin_model import block


def corrected_legs(source, upstream):
    edits = []
    joints = list(re.finditer(r'DEF KHR3HV_(\w+) HingeJoint\s*\{', source))
    if len(joints) != 22:
        raise ValueError('Expected 22 source joints')
    for match in joints:
        start = source.index('{', match.start()) + 1
        content = block(source, match.start())
        official_match = re.search(r'DEF KHR3HV_' + match.group(1) + r' HingeJoint\s*\{', upstream)
        if official_match is None:
            raise ValueError('Missing neutral reference: ' + match.group(1))
        official = block(upstream, official_match.start())
        endpoint = block(official, official.index('endPoint Solid')).split('children', 1)[0]
        begin = source.index('{', start + content.index('endPoint Solid')) + 1
        end = source.index('children', begin)
        edits.append((begin, end, endpoint))
        begin = source.index('{', start + content.index('jointParameters')) + 1
        end = source.index('}', begin)
        params = re.sub(r'\bposition\s+[-+0-9.eE]+', 'position 0', source[begin:end])
        edits.append((begin, end, params))
    for begin, end, text in sorted(edits, reverse=True):
        source = source[:begin] + text + source[end:]
    for name in ('LeftFemur', 'RightFemur', 'LeftCrus', 'RightCrus', 'RightAnkle'):
        pattern = r'(DEF KHR3HV_' + name + r' HingeJoint\s*\{\s*jointParameters HingeJointParameters\s*\{)([^}]*)(\})'
        def change(match):
            params = re.sub(r'\baxis\s+[-+0-9.eE ]+', '', match.group(2))
            return match.group(1) + '\n axis 0 1 0\n' + params + match.group(3)
        source, count = re.subn(pattern, change, source)
        if count != 1:
            raise ValueError('Missing axis: ' + name)
    return source


def build_world():
    here = Path(__file__).resolve().parent
    worlds = here.parents[1] / 'webots/kondo_RL/worlds'
    upstream = (here / 'runs/twin_free_space_20261023_01/upstream_Khr3hv_R2025a.proto').read_text(encoding='utf-8')
    source = (worlds / 'kondoKHR3-HV-box-slab-torque125.wbt').read_text(encoding='utf-8')
    from diagnose_joints import sensor_world
    content = sensor_world(corrected_legs(source, upstream))
    content, count = re.subn(r'(\bRobot\s*\{\s*)translation[^\n]*\n\s*rotation[^\n]*',
                            r'\1translation 0 0 0.24\n  rotation 0 0 1 0', content, count=1)
    assert count == 1
    assert 'gravity 0' not in content and 'selfCollision TRUE' in content
    assert 'controller "kondo_tcp_server"' in content
    assert content.count('maxTorque 1.716125') == 12
    world = worlds / 'kondoKHR3-HV-box-slab-corrected.wbt'
    world.write_text(content, encoding='utf-8')
    return world


if __name__ == '__main__':
    print(build_world())
