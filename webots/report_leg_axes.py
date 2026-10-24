"""Resolve nested joint axes into the neutral robot-body frame."""
import json
from pathlib import Path
import re
from audit_twin_model import block, vector, rotate, matmul


def report(path):
    text = path.read_text(encoding='utf-8')
    nodes = []
    for match in re.finditer(r'DEF KHR3HV_(\w+) HingeJoint\s*\{', text):
        content = block(text, match.start())
        params = block(content, content.index('jointParameters'))
        endpoint = block(content, content.index('endPoint Solid')).split('children', 1)[0]
        axis = vector(params, 'axis', [1., 0., 0.])
        q = vector(params, 'position', [0.])[0]
        rotation = vector(endpoint, 'rotation', [0., 0., 1., 0.])
        # Joint axis is expressed in parent coordinates: remove saved q on the left.
        rest = matmul(rotate(axis, -q), rotate(rotation[:3], rotation[3]))
        parent = next((n for n in reversed(nodes) if n['end'] > match.start()), None)
        matrix = parent['child_rotation'] if parent else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        resolved = [sum(matrix[i][j] * axis[j] for j in range(3)) for i in range(3)]
        nodes.append(dict(name=match.group(1), end=text.index('{', match.start())+len(content)+2,
                          local_axis=axis, body_axis=[round(v, 6) for v in resolved],
                          child_rotation=matmul(matrix, rest)))
    return [{k: n[k] for k in ('name', 'local_axis', 'body_axis')} for n in nodes
            if any(s in n['name'] for s in ('Femur', 'Crus', 'Ankle', 'Foot'))]


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    current = here.parents[1] / 'webots/kondo_RL/worlds/kondoKHR3-HV-box-slab-torque125-stepping-v3.wbt'
    official = here / 'runs/twin_free_space_20261023_01/upstream_Khr3hv_R2025a.proto'
    result = dict(current=report(current), official=report(official))
    output = here / 'runs/leg_axis_report_20261024.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
