"""Export the corrected expanded robot as a reusable local PROTO and sample world."""
from pathlib import Path
import re
from audit_twin_model import block


def main():
    here = Path(__file__).resolve().parent
    project = here.parents[1] / 'webots/kondo_RL'
    text = (project / 'worlds/kondoKHR3-HV-box-slab-corrected.wbt').read_text(encoding='utf-8')
    match = re.search(r'\bRobot\s*\{', text)
    body = 'Robot {' + block(text, match.start()) + '}'
    substitutions = [(r'\btranslation\s+0 0 0\.24', 'translation IS translation'),
        (r'\brotation\s+0 0 1 0', 'rotation IS rotation'),
        (r'\bcontroller\s+"kondo_tcp_server"', 'controller IS controller'),
        (r'\bsupervisor\s+TRUE', 'supervisor IS supervisor'),
        (r'\bselfCollision\s+TRUE', 'selfCollision IS selfCollision'),
        (r'\bname\s+"KHR-3HV"', 'name IS name')]
    for pattern, replacement in substitutions:
        body, count = re.subn(pattern, replacement, body, count=1)
        assert count == 1, pattern
    header = '''#VRML_SIM R2025a utf8
# Derived from Cyberbotics Khr3hv model; asset license retained.
# license: Copyright Cyberbotics Ltd. Licensed for use only with Webots.
# license url: https://cyberbotics.com/webots_assets_license
# Local corrections: neutral pose, five pitch axes, 25% leg torque, position sensors.
PROTO CorrectedKondoKHR3HV [
  field SFVec3f translation 0 0 0.24
  field SFRotation rotation 0 0 1 0
  field SFString name "Corrected Kondo KHR-3HV"
  field SFString controller "kondo_tcp_server"
  field SFBool supervisor TRUE
  field SFBool selfCollision TRUE
] {
'''
    proto = project / 'protos/CorrectedKondoKHR3HV.proto'
    proto.write_text(header + body + '\n}\n', encoding='utf-8')
    start = match.start()
    end = text.index('{', start) + len(block(text, start)) + 2
    sample = text[:start] + 'CorrectedKondoKHR3HV { }' + text[end:]
    sample = sample.replace('WorldInfo {', 'EXTERNPROTO "../protos/CorrectedKondoKHR3HV.proto"\n\nWorldInfo {', 1)
    world = project / 'worlds/kondo-corrected-proto-example.wbt'
    world.write_text(sample, encoding='utf-8')
    print(str(proto) + '\n' + str(world))


if __name__ == '__main__':
    main()
