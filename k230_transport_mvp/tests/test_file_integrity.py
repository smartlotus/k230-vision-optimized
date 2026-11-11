"""File-byte compatibility and bounded-memory regression tests; host only."""
import hashlib
from pathlib import Path
import tempfile
import tracemalloc
import unittest

from tools.file_integrity import sha256_file


class FileIntegrityTests(unittest.TestCase):
    def test_empty_binary_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            for size in (0, 1, 65535, 65536, 65537, 131079):
                with self.subTest(size=size):
                    data = (bytes(range(256)) * (size // 256 + 1))[:size]
                    path.write_bytes(data)
                    self.assertEqual(sha256_file(path), hashlib.sha256(data).hexdigest())
                    self.assertEqual(sha256_file(str(path)), hashlib.sha256(data).hexdigest())

    def test_rehash_detects_same_size_edit_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"original")
            before = sha256_file(path)
            path.write_bytes(b"modified")
            self.assertNotEqual(sha256_file(path), before)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                sha256_file(path)

    def test_large_file_uses_bounded_scratch_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            block = bytes(range(256)) * 4096
            expected = hashlib.sha256()
            with path.open("wb") as stream:
                for _ in range(16):
                    stream.write(block)
                    expected.update(block)
            tracemalloc.start()
            try:
                actual = sha256_file(path)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertEqual(actual, expected.hexdigest())
            self.assertLess(peak, 1024 * 1024)
