"""Compare the expanded active model with the versioned official Webots PROTO."""
import argparse
import json
import math
from pathlib import Path
import re


def block(text, start):
    begin = text.index('{', start)
    depth = 0
    for i in range(begin, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[begin + 1:i]
    raise ValueError('Unclosed node')


def vector(text, name, default):
    match = re.search(r'\b' + name + r'\s+([-+0-9.eE ]+)\s*(?:\n|$)', text)
    return list(map(float, match.group(1).split())) if match else default


def rotate(axis, angle):
    norm = math.sqrt(sum(v * v for v in axis))
    x, y, z = [v / norm for v in axis]
    c, s = math.cos(angle), math.sin(angle)
    return [[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def joints(text):
    result = {}
    for match in re.finditer(r'DEF KHR3HV_(\w+) HingeJoint\s*\{', text):
        content = block(text, match.start())
        params = block(content, content.index('jointParameters'))
        motor = block(content, content.index('RotationalMotor'))
        endpoint = block(content, content.index('endPoint Solid'))
        prefix = endpoint.split('children', 1)[0]
        axis = vector(params, 'axis', [1., 0., 0.])
        q = vector(params, 'position', [0.])[0]
        rotation = vector(prefix, 'rotation', [0., 0., 1., 0.])
        rest_rotation = matmul(rotate(rotation[:3], rotation[3]), rotate(axis, -q))
        result[match.group(1)] = dict(axis=axis, anchor=vector(params, 'anchor', [0., 0., 0.]),
            saved_position_rad=q, endpoint_translation=vector(prefix, 'translation', [0., 0., 0.]),
            rest_rotation=rest_rotation,
            motor={k: vector(motor, k, [None])[0] for k in ('maxTorque', 'maxVelocity', 'minPosition', 'maxPosition')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    active_text = (here.parents[1] / 'webots/kondo_RL/worlds/kondoKHR3-HV.wbt').read_text(encoding='utf-8')
    upstream_text = args.upstream.read_text(encoding='utf-8')
    current, upstream = joints(active_text), joints(upstream_text)
    comparisons = []
    for name in current.keys() | upstream.keys():
        if name not in current or name not in upstream:
            comparisons.append(dict(joint=name, missing=True))
            continue
        a, b = current[name], upstream[name]
        comparisons.append(dict(joint=name, axis_matches=a['axis'] == b['axis'],
            anchor_matches=a['anchor'] == b['anchor'], motor_matches=a['motor'] == b['motor'],
            endpoint_translation_matches=a['endpoint_translation'] == b['endpoint_translation'],
            zero_pose_matrix_max_error=max(abs(a['rest_rotation'][i][j] - b['rest_rotation'][i][j])
                                           for i in range(3) for j in range(3)), saved_position_rad=a['saved_position_rad']))
    def physics(text):
        return [[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', p)]
                for p in re.findall(r'physics Physics\s*\{([^{}]*)\}', text)]
    report = dict(joint_count=len(comparisons), joints=sorted(comparisons, key=lambda r: r['joint']),
                  physics_blocks_match=physics(active_text) == physics(upstream_text),
                  active_physics_count=len(physics(active_text)), upstream_physics_count=len(physics(upstream_text)),
                  active_contact_material_count=len(re.findall(r'\bcontactMaterial\b', active_text)))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
