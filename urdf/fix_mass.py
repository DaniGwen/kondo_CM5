import xml.etree.ElementTree as ET

# Load your current URDF
tree = ET.parse('khr.urdf')
root = tree.getroot()

# Inject a 100-gram dummy mass into every single limb
for link in root.findall('link'):
    inertial = ET.Element('inertial')
    ET.SubElement(inertial, 'mass', value='0.1')
    ET.SubElement(inertial, 'inertia', ixx='0.001', ixy='0', ixz='0', iyy='0.001', iyz='0', izz='0.001')
    link.append(inertial)

# Save the physics-ready URDF
tree.write('khr_physics.urdf', xml_declaration=True, encoding='utf-8')
print("Dummy mass injected! Saved as khr_physics.urdf")