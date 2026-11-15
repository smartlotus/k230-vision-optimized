"""Keep engineering fixtures separate from the frozen formal held-out test."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_saved_raw import evaluate_raw
from simulate_dataset import run


class SavedRawScopeTests(unittest.TestCase):
    def test_bounded_train_qualification_cannot_silently_read_test_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'bounded train engineering fixture'):
                run(root/'missing',root/'missing-model',root/'unused',split='synthetic_test',qualification_count=2)
            self.assertFalse((root/'unused').exists())

    def test_pilot_cannot_claim_the_formal_independent_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pred=root/'pred';export=root/'export';pred.mkdir();export.mkdir()
            (pred/'RESULT.json').write_text(json.dumps({'status':'NNCASE_AND_ORT_TEST_RAW_COMPLETE',
                'scope':'HELD_OUT_SYNTHETIC_TEST','split':'synthetic_test','images':1,'completed_images':1,
                'identity':{'onnx_sha256':'a'*64}}),encoding='utf8')
            (export/'RESULT.json').write_text(json.dumps({'onnx_sha256':'a'*64,'experiment':'pilot_pretrained'}),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'selected formal model'):
                evaluate_raw(root/'missing-data',pred,export,root/'unused.json')
            self.assertFalse((root/'unused.json').exists())


if __name__=='__main__':unittest.main()
