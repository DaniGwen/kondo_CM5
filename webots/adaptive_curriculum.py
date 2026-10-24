"""Evaluation gates for the mixed-push experiment (no simulator dependencies)."""
def key(row):
    return (row['force_N'], row['angle_deg'])


def rank(rows):
    return (sum(r['outcome'] == 'failed' for r in rows),
            sum(r['fell'] for r in rows),
            round(sum(r['peak_tilt_deg'] for r in rows) / len(rows), 1))


def preserves_reference(reference, candidate):
    outcomes = {key(r): r['outcome'] for r in candidate}
    return all(outcomes.get(key(r)) in ('recovered', 'no_measurable_disturbance')
               for r in reference if r['outcome'] != 'failed')


def can_advance(reference, candidate, gate_rows):
    # A threshold below 2 degrees counts as disturbance resistance, not recovery.
    passed = sum(r['outcome'] != 'failed' for r in gate_rows)
    measurable = sum(r['outcome'] == 'recovered' for r in gate_rows)
    return (preserves_reference(reference, candidate) and passed >= 3 and
            measurable >= 1 and len(gate_rows) == 4)
