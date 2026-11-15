"""Host-only transfer tests; the fake RPC executes the real file-assembly script."""
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_store import atomic_json, sha
from deploy_board_trial import CHUNK_BYTES, client_from_config, deploy, remote_jobs
from package_formal_delivery import BOARD_NAMES


class Fixture:
    def __init__(self, root):
        self.delivery = root / 'delivery'
        self.board = self.delivery / 'board'
        self.models = self.delivery / 'models'
        self.board.mkdir(parents=True)
        self.models.mkdir()
        model = bytes(range(251)) * 2400
        (self.models / 'topface.kmodel').write_bytes(model)
        identity = {'checkpoint_sha256': 'a' * 64,
                    'kmodel_sha256': sha(self.models / 'topface.kmodel')}
        atomic_json(self.models / 'identity.json', identity)
        atomic_json(self.models / 'output_contract.json', {'scope': 'fixture'})
        for name in BOARD_NAMES - {'MANIFEST.json'}:
            path = self.board / name
            if (self.models / name).exists():
                path.write_bytes((self.models / name).read_bytes())
            elif name == 'BENCHMARK_STATUS.json':
                atomic_json(path, {'status': 'NOT_RUN', 'model_sha256': identity['kmodel_sha256']})
            else:
                path.write_text('fixture\n', encoding='utf8')
        def rows():
            return [{'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                    for path in sorted(self.board.iterdir()) if path.name in BOARD_NAMES]
        atomic_json(self.board / 'MANIFEST.json',
                    {'status': 'DEPLOYMENT_BUNDLE_READY', 'board_test': 'NOT_RUN', 'files': rows()})
        atomic_json(self.board / 'MEMBERS.json', rows())
        self.inventory = {'status': 'FORMAL_BOARD_TEST_PACKAGE_VERIFIED', 'run_id': 'fixture-formal',
                          'checkpoint_sha256': identity['checkpoint_sha256'], 'native_board_test': 'NOT_RUN'}
        self.rehash()

    def rehash(self):
        self.inventory['files'] = [{'path': path.relative_to(self.delivery).as_posix(),
                                   'bytes': path.stat().st_size, 'sha256': sha(path)}
                                  for path in sorted(self.delivery.rglob('*'))
                                  if path.is_file() and path.name != 'SHA256SUMS.json']
        atomic_json(self.delivery / 'SHA256SUMS.json', self.inventory)


class FakeClient:
    def __init__(self, disk):
        self.disk = disk
        self.calls = []
        self.connected, self.running, self.exists = True, False, False
        self.terminal = 'PRESERVED_OLD_OUTPUT\n'
        self.corrupt_part, self.corrupt_target, self.fail_write = False, False, False
        self.stuck_join = False

    def local(self, path):
        self.assert_remote(path)
        return self.disk / path.removeprefix('/sdcard/')

    @staticmethod
    def assert_remote(path):
        if not path.startswith('/sdcard/tf_') or '..' in path:
            raise AssertionError('Unexpected mutation path')

    def tool(self, name, args=None):
        args = args or {}
        self.calls.append((name, args))
        if name == 'canmv_board_info':
            return {'connected': self.connected}
        if name == 'canmv_script_running':
            return {'running': self.running}
        if name == 'canmv_stat_file':
            return {'exists': self.exists}
        if name == 'canmv_mkdir':
            self.local(args['path']).mkdir()
            return {'success': True}
        if name == 'canmv_write_file':
            if self.fail_write:
                raise TimeoutError('Uncertain write response')
            self.local(args['path']).write_bytes(base64.b64decode(args['content'], validate=True))
            return {'success': True}
        if name == 'canmv_read_file':
            raw = self.local(args['path']).read_bytes()
            if self.corrupt_part:
                raw += b'tampered'
            return {'content': base64.b64encode(raw).decode('ascii')}
        if name == 'canmv_run_script':
            script = args['script'].replace('/sdcard/', self.disk.as_posix() + '/')
            output = io.StringIO()
            real_open = open
            def disk_open(path, mode='r', *positional, **keyword):
                if self.corrupt_target and mode == 'rb' and '.p' not in Path(path).name:
                    return io.BytesIO(b'bad stored bytes')
                return real_open(path, mode, *positional, **keyword)
            try:
                with redirect_stdout(output):
                    exec(compile(script, '<board-file-assembly>', 'exec'), {'open': disk_open})
            except ValueError:
                pass  # Native submission can succeed while its script fails.
            self.terminal += output.getvalue().replace(self.disk.as_posix() + '/', '/sdcard/')
            if self.stuck_join:
                self.running = True
            return {'success': True}
        if name == 'canmv_terminal_output':
            assert args['clear'] is False
            return {'text': self.terminal}
        raise AssertionError(name)


class BoardDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        self.disk = self.root / 'sd'
        self.disk.mkdir()
        self.client = FakeClient(self.disk)
        self.receipt = self.root / 'install.json'

    def run_deploy(self, install=True):
        return deploy(self.fixture.delivery, '/sdcard/tf_a01', self.receipt,
                      install=install, client=self.client)

    def test_default_preflight_has_no_board_mutations(self):
        result = self.run_deploy(install=False)
        self.assertEqual(result['status'], 'PREFLIGHT_PASS')
        self.assertEqual([name for name, _ in self.client.calls],
                         ['canmv_board_info', 'canmv_script_running', 'canmv_stat_file'])
        self.assertEqual(list(self.disk.iterdir()), [])

    def test_real_join_roundtrip_and_multichunk_model(self):
        result = self.run_deploy()
        self.assertEqual(result['status'], 'VERIFIED_BOARD_FILES_ONLY')
        self.assertFalse(result['inference_started'])
        self.assertEqual(result['native_fps'], 'NOT_RUN')
        model = next(job for job in result['files'] if job['target'].endswith('/topface.kmodel'))
        self.assertGreater(len(model['parts']), 1)
        self.assertTrue(all(row['bytes'] <= CHUNK_BYTES for row in result['verified_parts']))
        self.assertTrue(self.client.terminal.startswith('PRESERVED_OLD_OUTPUT\n'))
        for name in BOARD_NAMES:
            self.assertEqual((self.fixture.board / name).read_bytes(),
                             (self.disk / 'tf_a01' / name).read_bytes())
        run_calls = [args for name, args in self.client.calls if name == 'canmv_run_script']
        self.assertEqual(len(run_calls), 1)
        self.assertNotIn('topface_trial import', run_calls[0]['script'])

    def test_connected_idle_and_new_directory_required(self):
        for field, value in (('connected', False), ('running', True), ('exists', True), ('exists', None)):
            with self.subTest(field=field, value=value):
                client = FakeClient(self.disk)
                setattr(client, field, value)
                with self.assertRaises(RuntimeError):
                    deploy(self.fixture.delivery, '/sdcard/tf_a01', self.receipt, install=True, client=client)
                self.assertFalse(any(name in ('canmv_mkdir', 'canmv_write_file', 'canmv_run_script')
                                     for name, _ in client.calls))

    def test_tampered_model_stops_before_rpc(self):
        (self.fixture.board / 'topface.kmodel').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.run_deploy()
        self.assertEqual(self.client.calls, [])

    def test_missing_board_inventory_stops_before_rpc(self):
        self.fixture.inventory['files'] = [row for row in self.fixture.inventory['files']
                                           if row['path'] != 'board/topface_trial.py']
        atomic_json(self.fixture.delivery / 'SHA256SUMS.json', self.fixture.inventory)
        with self.assertRaises(ValueError):
            self.run_deploy()
        self.assertEqual(self.client.calls, [])

    def test_part_corruption_prevents_join_and_is_not_retried(self):
        self.client.corrupt_part = True
        with self.assertRaises(RuntimeError):
            self.run_deploy()
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result['status'], 'NEEDS_RECONCILIATION')
        self.assertFalse(any(name == 'canmv_run_script' for name, _ in self.client.calls))
        self.assertEqual(sum(name == 'canmv_write_file' for name, _ in self.client.calls), 1)

    def test_stored_target_is_reopened_and_verified(self):
        self.client.corrupt_target = True
        with self.assertRaises(RuntimeError):
            self.run_deploy()
        self.assertEqual(json.loads(self.receipt.read_text())['status'], 'NEEDS_RECONCILIATION')

    def test_uncertain_write_retains_receipt_without_retry(self):
        self.client.fail_write = True
        with self.assertRaises(RuntimeError):
            self.run_deploy()
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result['status'], 'NEEDS_RECONCILIATION')
        self.assertEqual(result['current_part'], '/sdcard/tf_a01/BENCHMARK_STATUS.json.p0')
        self.assertEqual(sum(name == 'canmv_write_file' for name, _ in self.client.calls), 1)

    def test_join_timeout_preserves_running_script_without_resubmission(self):
        self.client.stuck_join = True
        with patch('deploy_board_trial.time.monotonic', side_effect=[0, 31]):
            with self.assertRaises(RuntimeError):
                self.run_deploy()
        result = json.loads(self.receipt.read_text())
        self.assertEqual(result['status'], 'NEEDS_RECONCILIATION')
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertTrue(self.client.running)
        self.assertEqual(sum(name == 'canmv_run_script' for name, _ in self.client.calls), 1)
        self.assertFalse(any('stop' in name for name, _ in self.client.calls))

    def test_existing_receipt_and_production_paths_are_preserved(self):
        self.receipt.write_text('preserve')
        with self.assertRaises(ValueError):
            self.run_deploy()
        self.assertEqual(self.receipt.read_text(), 'preserve')
        for remote in ('/sdcard', '/sdcard/main.py', '/sdcard/topface_trial', '/sdcard/tf_a/../boot.py'):
            with self.subTest(remote=remote), self.assertRaises(ValueError):
                remote_jobs(remote, [('topface_trial.py', b'fixture')])

    def test_inventory_traversal_is_rejected_before_rpc(self):
        self.fixture.inventory['files'][0]['path'] = '../outside'
        atomic_json(self.fixture.delivery / 'SHA256SUMS.json', self.fixture.inventory)
        with self.assertRaises(ValueError):
            self.run_deploy()
        self.assertEqual(self.client.calls, [])

    def test_preview_requires_explicit_scope_and_completed_training(self):
        self.fixture.inventory.update(status='FORMAL_BOARD_PREVIEW_PACKAGE_VERIFIED',
                                      scope='FORMAL_BOARD_PREVIEW_ONLY', full_934_test='PENDING_ORIGINAL_07B',
                                      training_completed_epochs=100)
        self.fixture.rehash()
        self.assertEqual(self.run_deploy(install=False)['status'], 'PREFLIGHT_PASS')
        self.receipt.unlink()
        self.fixture.inventory['training_completed_epochs'] = 99
        self.fixture.rehash()
        self.client.calls.clear()
        with self.assertRaises(ValueError):
            self.run_deploy()
        self.assertEqual(self.client.calls, [])

    def test_explicit_config_rejects_remote_or_disabled_service(self):
        for url, enabled in (('https://127.0.0.1:1234/mcp', True),
                             ('http://example.invalid/mcp', True),
                             ('http://127.0.0.1:1234/mcp?extra=1', True),
                             ('http://127.0.0.1:1234/mcp', False)):
            with self.subTest(url=url, enabled=enabled):
                config = self.root / 'mcp.toml'
                config.write_text('[mcp_servers.canmv-k230]\nurl=' + json.dumps(url) + '\nenabled=' +
                                  str(enabled).lower() + '\n', encoding='utf8')
                with patch('urllib.request.build_opener', side_effect=AssertionError('No network')):
                    with self.assertRaises(ValueError):
                        client_from_config(config)

    def test_explicit_config_requires_authentication(self):
        config = self.root / 'mcp.toml'
        config.write_text('[mcp_servers.canmv-k230]\nurl="http://127.0.0.1:1234/mcp"\n', encoding='utf8')
        with patch('urllib.request.build_opener', side_effect=AssertionError('No network')):
            with self.assertRaises(ValueError):
                client_from_config(config)

    def test_explicit_config_initializes_existing_single_send_client(self):
        config = self.root / 'mcp.toml'
        config.write_text('[mcp_servers.canmv-k230]\nurl="http://127.0.0.1:1234/mcp"\n'
                          '[mcp_servers.canmv-k230.http_headers]\nAuthorization="fixture-only"\n',
                          encoding='utf8')
        # Import through the same explicit module root used by the factory.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'k230_transport_mvp'))
        from tools.deploy_native_trial import CanMvClient
        with patch.object(CanMvClient, 'request', return_value={}) as request:
            client = client_from_config(config)
        self.assertIsInstance(client, CanMvClient)
        self.assertEqual(client.url, 'http://127.0.0.1:1234/mcp')
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0], 'initialize')


if __name__ == '__main__':
    unittest.main()
