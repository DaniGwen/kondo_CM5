import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile
from export_readiness import assess, sha256


class ReadinessTests(unittest.TestCase):
    def setup_run(self, folder, count=66, steps=0):
        run = folder / 'candidate'
        run.mkdir()
        with zipfile.ZipFile(run / 'best_model.zip', 'w') as archive:
            archive.writestr('data', json.dumps(dict(num_timesteps=steps, observation_space={'_shape':[count]})))
        (run / 'experiment.json').write_text(json.dumps(dict(algorithm='TD3', world_version='corrected-model')))
        runner = folder / 'runner.py'
        runner.write_text('SERVO_IDS = ' + repr(list(range(14))) + '\nSTANDING_ANGLES_DEG = ' + repr([0]*14))
        return run, runner

    def test_simulator_model_is_blocked(self):
        with TemporaryDirectory() as temporary:
            run, runner = self.setup_run(Path(temporary))
            result = assess(run, runner)
            self.assertFalse(result['hardware_ready'])
            self.assertTrue(any('66 features' in s for s in result['blockers']))
            self.assertTrue(any('initial baseline' in s for s in result['blockers']))

    def test_approval_requires_checkpoint_hash_and_evidence(self):
        with TemporaryDirectory() as temporary:
            run, runner = self.setup_run(Path(temporary), count=6, steps=100000)
            keys = ('nominal_torque_passed','sensor_contract_passed','servo_mapping_passed','supported_dry_run_passed','recovery_suite_passed')
            report = dict(model_sha256=sha256(run / 'best_model.zip'), evidence={})
            report.update({key:True for key in keys})
            (run / 'deployment_validation.json').write_text(json.dumps(report))
            self.assertFalse(assess(run, runner)['hardware_ready'])
            for key in keys:
                path = run / (key + '.json')
                path.write_text('{}')
                report['evidence'][key] = dict(file=path.name, sha256=sha256(path))
            (run / 'deployment_validation.json').write_text(json.dumps(report))
            self.assertTrue(assess(run, runner)['hardware_ready'])
            report['model_sha256'] = 'wrong-model'
            (run / 'deployment_validation.json').write_text(json.dumps(report))
            self.assertFalse(assess(run, runner)['hardware_ready'])


if __name__ == '__main__':
    unittest.main()
