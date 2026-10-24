"""Replace the current arena floor with a static slab, preserving wall geometry."""
import re


def scale_leg_torque(source, scale):
    if not 1 <= scale <= 2:
        raise ValueError('Expected torque scale between 1 and 2')
    names = {side + joint for side in ('Left', 'Right')
             for joint in ('FemurHead1', 'FemurHead2', 'Femur', 'Crus', 'Ankle', 'Foot')}
    found = set()
    def change(match):
        text = match.group(0)
        name = re.search(r'name\s+"([^"]+)"', text)
        if name and name.group(1) in names:
            found.add(name.group(1))
            text, count = re.subn(r'maxTorque\s+([0-9.]+)',
                lambda value: 'maxTorque ' + format(float(value.group(1)) * scale, '.6f'), text)
            if count != 1:
                raise ValueError('Expected one torque limit per leg motor')
        return text
    result = re.sub(r'RotationalMotor\s*\{[^}]*\}', change, source)
    if found != names:
        raise ValueError('Expected all twelve leg motors')
    return result


def use_box_slab(source):
    def solid(name, position, size, color):
        return f'''Solid {{
  translation {position}
  name "{name}"
  contactMaterial "default"
  children [ Shape {{
    appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }}
    geometry DEF {name}Geometry Box {{ size {size} }}
  }} ]
  boundingObject USE {name}Geometry
  locked TRUE
}}
'''
    # RectangleArena defaults: 1x1 m floor, 0.01 m walls, 0.1 m wall height.
    # Floor top stays at z=-0.1; the 0.1 m slab extends downward.
    replacement = solid('BoxSlab', '0 0 -0.15', '1 1 0.1', '0.5 0.5 0.5')
    for name, position, size in [
        ('WallRight', '0.505 0 -0.05', '0.01 1 0.1'),
        ('WallLeft', '-0.505 0 -0.05', '0.01 1 0.1'),
        ('WallBack', '0 0.505 -0.05', '1.02 0.01 0.1'),
        ('WallFront', '0 -0.505 -0.05', '1.02 0.01 0.1')]:
        replacement += solid(name, position, size, '0.3 0.3 0.3')
    result, count = re.subn(r'RectangleArena\s*\{\s*translation\s+0\s+0\s+-0\.1\s*\}', replacement, source)
    if count != 1:
        raise ValueError('Expected the original default-size arena at z=-0.1')
    return result
