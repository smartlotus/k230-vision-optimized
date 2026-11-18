"""Preserve bytes and evidence when streaming into limited board storage."""
import contextlib
import ast
import io
from pathlib import Path
import tempfile
import unittest

from native_smoke import append_script, idle, remote_root, runtime_append_script


class StreamingInstallTests(unittest.TestCase):
    def test_canmv_base64_decoder_receives_bytes(self):
        script = runtime_append_script('/tmp/k230_temporal_v1_test/model', 0, b'data', 'CHECK:')
        calls = [node for node in ast.walk(ast.parse(script))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'a2b_base64']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(ast.literal_eval(calls[0].args[0]), bytes)

    def execute(self, part, target, offset, data):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(append_script(str(part), str(target), offset, data, 'CHECK:'), {})

    def test_chunk_order_and_verified_part_reclamation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            part = Path(directory) / 'part'
            expected = b''
            for data in (bytes(range(256)) * 1024, b'final-short-part'):
                part.write_bytes(data)
                self.execute(part, target, len(expected), data)
                expected += data
                self.assertEqual(target.read_bytes(), expected)
                self.assertFalse(part.exists())

    def test_changed_part_is_preserved_without_appending(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            part = Path(directory) / 'part'
            target.write_bytes(b'old-prefix')
            part.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Part identity differs'):
                self.execute(part, target, 10, b'expected')
            self.assertEqual(target.read_bytes(), b'old-prefix')
            self.assertEqual(part.read_bytes(), b'corrupt')

    def test_wrong_offset_preserves_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            part = Path(directory) / 'part'
            target.write_bytes(b'existing')
            part.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'Unexpected assembly offset'):
                self.execute(part, target, 0, b'new')
            self.assertEqual(target.read_bytes(), b'existing')
            self.assertEqual(part.read_bytes(), b'new')

    def test_test_roots_exclude_system_and_other_experiments(self):
        self.assertEqual(remote_root('/tmp/k230_temporal_v1_a'), '/tmp/k230_temporal_v1_a')
        self.assertEqual(remote_root('/data/k230_temporal_v1_a'), '/data/k230_temporal_v1_a')
        for path in ('/bin/k230_temporal_v1_a', '/sdcard/tfp1116', '/tmp/xyusb1117e',
                     '/tmp/k230_temporal_v1_a/../xyusb1117f'):
            with self.assertRaises(ValueError):
                remote_root(path)

    def test_busy_or_unknown_board_cannot_start_an_install(self):
        for result in ({'running': True}, {}):
            client = type('Client', (), {'tool': lambda _, name, r=result: r})()
            with self.assertRaisesRegex(RuntimeError, 'Board busy or unknown'):
                idle(client)


if __name__ == '__main__':
    unittest.main()
