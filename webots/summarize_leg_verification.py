import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here / 'runs'
datasets = {name: json.loads((root / ('leg_joint_verification_20261024_' + name) / 'measured_axes.json').read_text())
            for name in ('original', 'candidate_neutral')}
for variant, rows in datasets.items():
    assert len(rows) == 48
    for row in rows:
        assert abs(row['measured_rad'] - row['command_rad']) < 1e-6
        if variant == 'candidate_neutral':
            name = row['joint']
            expected = [0, 0, -1 if name.startswith('Right') else 1] if name.endswith('FemurHead1') else [
                -1 if name.startswith('Right') else 1, 0, 0] if name.endswith(('FemurHead2', 'Foot')) else [0, 1, 0]
            assert max(abs(a-b) for a, b in zip(row['measured_body_axis'], expected)) < 1e-6
lines = ['# All-leg-joint verification', '',
         '96 isolated sweep samples: 12 joints, four nonzero angles each (-0.25, -0.1, +0.1, +0.25 rad), two model variants. Neutral references acquired for every joint. Zero gravity, root clear of floor, self-collision disabled. Position sensors verify commanded coordinates; foot orientation changes verify the effective body-frame rotation axis, including foot joints whose origin does not translate.', '',
         '| Joint | Original axis | Candidate axis | Intended role |', '|---|---|---|---|']
for row in datasets['original']:
    if row['command_rad'] != .1:
        continue
    name = row['joint']
    candidate = next(r for r in datasets['candidate_neutral'] if r['joint'] == name and r['command_rad'] == .1)
    fmt = lambda seq: ', '.join(str(round(v)) for v in seq)
    role = 'hip yaw' if name.endswith('FemurHead1') else 'hip roll' if name.endswith('FemurHead2') else 'hip pitch' if name.endswith('Femur') else 'knee pitch' if name.endswith('Crus') else 'ankle pitch' if name.endswith('Ankle') else 'ankle roll'
    lines.append(f"| {name} | {fmt(row['measured_body_axis'])} | {fmt(candidate['measured_body_axis'])} | {role} |")
lines += ['', '## Results', '',
    'Five pitch axes are inconsistent with the expected sagittal hinge motion: RightFemur, LeftFemur, RightCrus, LeftCrus, RightAnkle. LeftAnkle already rotates about body Y. Hip yaw, hip roll, and ankle roll behave consistently with their intended axis categories.', '',
    'The candidate sets the five pitch axes to local (0,1,0). All 48 candidate sweep samples match expected body-frame axes within 1e-6, and all 96 measured joint coordinates match commands within 1e-6 rad. Positive pitch rotates both legs toward negative body X: +0.1 rad produces approximately -16.48 mm foot displacement at either hip pitch, -10.11 mm at either knee, and -3.69 mm at either ankle pitch, with negligible lateral movement. Both command signs and two magnitudes were verified. No motor index swap is required to obtain these motions.', '',
    'The source world contains saved nonzero joint coordinates and corresponding endpoint rotations. A first axis-only candidate retained incompatible saved rotations and produced skewed downstream axes. The validated candidate instead uses neutral endpoint transforms from the stored versioned stock source, sets saved coordinates to zero, and then changes pitch axes. Changing only the axis fields in the saved pose is insufficient.', '',
    'This verifies isolated simulator kinematics, not motor strength, contact behavior, physical servo IDs, safe hardware travel, or real servo signs. The candidate remains a diagnostic zero-gravity/no-self-collision world and is not a ready training world. A production candidate must restore gravity/collisions, retain box-slab/torque settings, and pass neutral holds and mirrored small-motion tests before training.', '',
    'TD3 remains stopped with its model/replay saved at 25001 steps. Existing training worlds and old policy artifacts remain unchanged. Replay from the old kinematics is incompatible with the corrected model.', '',
    'Raw results: leg_joint_verification_20261024_original/measured_axes.json and leg_joint_verification_20261024_candidate_neutral/measured_axes.json.']
output = root / 'leg_joint_verification_20261024_candidate_neutral/verification.md'
output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('PASS: 96 position checks and 48 corrected-axis checks. Report: ' + str(output))
