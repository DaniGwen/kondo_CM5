import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import training_ui as ui


class DashboardTests(unittest.TestCase):
    def config(self):
        return dict(name='ui_test', policy='TD3', steps=2000000, learning_rate=.0001,
                    rewards=dict(torque=.75, saturation=.75, tracking=.25, step=.75))

    def test_validation(self):
        self.assertEqual(ui.validate_config(self.config())['steps'], 2000000)
        for key, value in [('name', '../bad'), ('policy', 'OTHER'), ('steps', 1), ('learning_rate', float('nan'))]:
            data = self.config()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ui.validate_config(data)
        data = self.config()
        data['rewards']['step'] = float('inf')
        with self.assertRaises(ValueError):
            ui.validate_config(data)

    def test_existing_training_cannot_be_overwritten(self):
        with TemporaryDirectory() as temporary, patch.object(ui, 'RUNS', Path(temporary)), patch.object(ui, 'ports_busy', return_value=True):
            server = ThreadingHTTPServer(('127.0.0.1', 0), ui.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = 'http://127.0.0.1:' + str(server.server_port)
                headers = {'Host': '127.0.0.1:8765', 'Content-Type': 'application/json', 'X-Training-Token': ui.TOKEN}
                request = Request(url + '/api/start', json.dumps(self.config()).encode(), headers=headers)
                with self.assertRaises(HTTPError) as result:
                    urlopen(request)
                self.assertEqual(result.exception.code, 400)
                self.assertFalse(list(Path(temporary).iterdir()))
                headers.pop('X-Training-Token')
                with self.assertRaises(HTTPError) as result:
                    urlopen(Request(url+'/api/stop', b'{}', headers=headers))
                self.assertEqual(result.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
