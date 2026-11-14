import json
import unittest
from unittest.mock import patch
from publisher.control import validate_binding


class PublisherBindingTests(unittest.TestCase):
    def test_protected_payload_cannot_substitute_reserved_drive_identity(self):
        binding={'run_id':'k230-bound-fixture-a01','archive':{'sha256':'a'*64},'prep':{'version':3},
            'folder_id':'1MwltNmaI24HjvcxnG1w608qui42RAPKf','drive_file_id':'reserved-file',
            'secret_name':'TOPFACE_TRANSFER_123456789','branch':'ops/topface-transfer-123456789',
            'head':'b'*40,'publisher_source_commit':'c'*40}
        protected={k:binding[k] for k in ('run_id','archive','prep','folder_id','drive_file_id')}
        protected['drive_file_id']='substituted-file'
        with patch('publisher.control.command') as command:
            with self.assertRaisesRegex(ValueError,'envelope differs'):
                validate_binding(binding,protected)
            command.assert_not_called()

    def test_head_message_and_parent_must_match_original_transfer(self):
        binding={'folder_id':'1MwltNmaI24HjvcxnG1w608qui42RAPKf',
            'secret_name':'TOPFACE_TRANSFER_123456789','branch':'ops/topface-transfer-123456789',
            'head':'b'*40,'publisher_source_commit':'c'*40}
        with patch('publisher.control.command',return_value='other-message'):
            with self.assertRaisesRegex(ValueError,'message differs'):validate_binding(binding)
        with patch('publisher.control.command',side_effect=['123456789','b'*40+' '+'d'*40]):
            with self.assertRaisesRegex(ValueError,'parent differs'):validate_binding(binding)


if __name__=='__main__':unittest.main()
