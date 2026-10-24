"""Exercise the actual TCP controller loop with a deterministic simulator stub."""
import contextlib
import io
import math
from pathlib import Path
import runpy
import struct
import sys
import types
import unittest
import tempfile
from unittest.mock import patch
from launch_finetune_balance import prepare_run


class Node:
    def __init__(self):
        self.forces = []
        self.restores = 0

    def saveState(self, name):
        pass

    def loadState(self, name):
        self.restores += 1

    def getField(self, name):
        return types.SimpleNamespace(setSFVec3f=lambda value: None,
                                     setSFRotation=lambda value: None)

    def resetPhysics(self):
        pass

    def getOrientation(self):
        return [1, 0, 0, 0, 1, 0, 0, 0, math.cos(math.radians(1))]

    def getPosition(self):
        return [0, 0, .24]

    def addForce(self, force, relative):
        self.forces.append(force)


class ServerTests(unittest.TestCase):
    def test_run_preserves_reference_and_starts_without_replay(self):
        here = Path(__file__).resolve().parent
        server = here.parents[1] / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py'
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.zip'
            source.write_bytes(b'reference checkpoint')
            run = Path(temporary) / 'run'
            prepare_run(source, run, {'KONDO_TOTAL_TIMESTEPS': '500000'}, here, server)
            self.assertEqual((run / 'reference_model.zip').read_bytes(), source.read_bytes())
            self.assertEqual((run / 'kondo_td3_v2.zip').read_bytes(), source.read_bytes())
            self.assertFalse((run / 'kondo_td3_v2_replay_buffer.pkl').exists())
            with self.assertRaises(FileExistsError):
                prepare_run(source, run, {'KONDO_TOTAL_TIMESTEPS': '500000'}, here, server)

    def test_scheduled_push_duration_ack_and_reset(self):
        node = Node()
        device = types.SimpleNamespace(enable=lambda dt: None,
                                       getValues=lambda: [0, 0, 0],
                                       setPosition=lambda value: None)
        robot = types.SimpleNamespace(getSelf=lambda: node, getDevice=lambda name: device,
                                      step=lambda dt: 0)
        reset = [999.] * 14
        reset[1:6] = [1, 2, 90, 5, 2]
        packets = [struct.pack('<14d', *reset)]
        packets += [struct.pack('<14d', *([0.] * 14))] * 130
        packets += [struct.pack('<14d', *reset)]
        connection = types.SimpleNamespace(recv=lambda n: packets.pop(0) if packets else b'',
                                           sendall=lambda data: responses.append(struct.unpack('<8d', data)),
                                           setsockopt=lambda *a: None, close=lambda: None)
        listener = types.SimpleNamespace(setsockopt=lambda *a: None, bind=lambda *a: None,
                                         listen=lambda *a: None, accept=lambda: (connection, 'test'),
                                         close=lambda: None)
        responses = []
        log = io.StringIO()
        path = Path(__file__).resolve().parents[2] / 'webots/kondo_RL/controllers/kondo_tcp_server/kondo_tcp_server.py'
        with patch.dict(sys.modules, {'controller': types.SimpleNamespace(Supervisor=lambda: robot)}), \
                patch('socket.socket', return_value=listener), \
                patch.dict('os.environ', {}, clear=True), contextlib.redirect_stdout(log):
            namespace = runpy.run_path(str(path), run_name='__main__')
        self.assertEqual(len(node.forces), 5)
        self.assertAlmostEqual(node.forces[0][1], 2)
        self.assertEqual(node.restores, 2)
        self.assertEqual(responses[0][5], -999.)
        self.assertEqual(responses[1][5], 0.)
        self.assertIn('duration_ms=80 impulse_Ns=0.160', log.getvalue())
        self.assertIn('outcome=no_measurable_disturbance', log.getvalue())
        reset_angles = []
        robot.getFromDef = lambda name: types.SimpleNamespace(setJointPosition=reset_angles.append)
        with patch.dict('os.environ', {'KONDO_RESET_JOINTS': '1'}), contextlib.redirect_stdout(log):
            namespace['reset_episode'](connection, 3)
        self.assertEqual(reset_angles, [0.0] * 22)


if __name__ == '__main__':
    unittest.main()
