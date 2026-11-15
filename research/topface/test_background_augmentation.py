"""Prevent held-out images and approved target pixels entering the negative bank."""
import json
from pathlib import Path
import tempfile
import unittest

from background_augmentation import BackgroundBank


class BackgroundProvenanceTests(unittest.TestCase):
    def manifest(self,source_split='train',boxes=None):
        return {'source_scope':'APPROVED_REAL_TRAIN_ONLY','source_manifest_sha256':'a'*64,
            'source_exclusion_margin_px':12,
            'sources':[{'image_id':'source','split':source_split,'approved_top_boxes_xyxy':boxes or []}],
            'members':[{'source_image_id':'source','source_xyxy':[0,0,96,96],'name':'patch-001.png'}]}

    def test_validation_and_test_sources_are_rejected_before_pixel_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for split in ('val','test'):
                (path/'MANIFEST.json').write_text(json.dumps(self.manifest(split)),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'Validation or test'):
                    BackgroundBank(path)

    def test_crop_overlapping_a_reviewed_top_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'MANIFEST.json').write_text(json.dumps(self.manifest(boxes=[[50,50,60,60]])),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'intersects an approved source target'):
                BackgroundBank(path)


if __name__=='__main__':unittest.main()
