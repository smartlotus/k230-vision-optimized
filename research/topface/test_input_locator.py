import hashlib
from pathlib import Path
import tempfile
import unittest
from input_locator import locate_input


class InputLocatorTests(unittest.TestCase):
    def test_old_and_namespaced_mounts_and_hash_mismatch(self):
        body=b'{"fixture":"frozen input"}'
        digest=hashlib.sha256(body).hexdigest()
        for layout in ('dataset-slug','datasets/owner/dataset-slug'):
            with self.subTest(layout=layout),tempfile.TemporaryDirectory() as temporary:
                mount=Path(temporary);target=mount/layout;target.mkdir(parents=True)
                (target/'INPUT_MANIFEST.json').write_bytes(body)
                self.assertEqual(locate_input(mount,digest),target.resolve())
                with self.assertRaises(RuntimeError):locate_input(mount,'0'*64)

    def test_duplicate_matching_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            mount=Path(temporary)
            for name in ('one','two'):
                directory=mount/name;directory.mkdir()
                (directory/'INPUT_MANIFEST.json').write_bytes(b'fixture')
            with self.assertRaises(RuntimeError):locate_input(mount,hashlib.sha256(b'fixture').hexdigest())


if __name__=='__main__':unittest.main()
