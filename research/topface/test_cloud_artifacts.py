import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from cloud_artifacts import extract_verified_archive


class CloudArtifactTests(unittest.TestCase):
    def archive(self,path,members):
        with tarfile.open(path,'w:gz') as bundle:
            for name,data in members:
                item=tarfile.TarInfo(name);item.size=len(data);bundle.addfile(item,io.BytesIO(data))

    def test_complete_member_inventory_and_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=b'original payload'
            manifest=json.dumps([{'name':'nested/data','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]).encode()
            self.archive(root/'good.tar.gz',[('nested/data',data),('MEMBERS.json',manifest)])
            result=extract_verified_archive(root/'good.tar.gz',root/'good')
            self.assertEqual(result['sha256_verification'],'ALL_MEMBERS_PASS')
            self.archive(root/'changed.tar.gz',[('nested/data',b'modified payload'),('MEMBERS.json',manifest)])
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                extract_verified_archive(root/'changed.tar.gz',root/'changed')

    def test_path_traversal_and_missing_inventory_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.archive(root/'unsafe.tar.gz',[('../outside',b'x'),('MEMBERS.json',b'[]')])
            with self.assertRaisesRegex(ValueError,'unsafe member'):
                extract_verified_archive(root/'unsafe.tar.gz',root/'unsafe')
            self.archive(root/'unlisted.tar.gz',[('extra',b'x'),('MEMBERS.json',b'[]')])
            with self.assertRaisesRegex(ValueError,'denominator'):
                extract_verified_archive(root/'unlisted.tar.gz',root/'unlisted')


if __name__=='__main__':unittest.main()
