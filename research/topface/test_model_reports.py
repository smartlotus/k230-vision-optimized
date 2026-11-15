"""Bounded evidence collection and true-negative failure-example regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from cloud_store import atomic_json
from cloud_store import sha
from collect_model_reports import CORE,manifest,validate_reports
from failure_previews import compare_frame,render
from simulate_dataset import order_sha
import cv2
import numpy as np


class ModelReportTests(unittest.TestCase):
    def saved(self):
        return {'export':{'experiment':'formal_full'},
                'evaluation_reports':[{'path':name,'bytes':100,'sha256':'a'*64} for name in sorted(CORE)]}

    def test_report_inventory_cannot_collect_raw_arrays_or_traversal_or_unbounded_files(self):
        manifest(self.saved())
        for name,size in [('paired-synthetic-test/simulator-raw.npy',255000),('../secret.json',20),('SYNTHETIC_TEST.json',3*1024**2)]:
            saved=self.saved();saved['evaluation_reports'].append({'path':name,'bytes':size,'sha256':'a'*64})
            with self.assertRaises(ValueError):manifest(saved)

    def test_pilot_cannot_claim_final_report_collection(self):
        saved=self.saved();saved['export']['experiment']='pilot_pretrained'
        with self.assertRaisesRegex(ValueError,'frozen formal'):manifest(saved)

    def test_failure_images_need_their_evidence_and_at_most_twelve_members(self):
        saved=self.saved()
        saved['evaluation_reports'].append({'path':'failure-previews/case-01.jpg','bytes':100,'sha256':'a'*64})
        with self.assertRaisesRegex(ValueError,'inventory'):manifest(saved)
        saved=self.saved();saved['evaluation_reports'].append({'path':'failure-previews/EVIDENCE.json','bytes':100,'sha256':'a'*64})
        saved['evaluation_reports'] += [{'path':f'failure-previews/case-{i:02d}.jpg','bytes':100,'sha256':'a'*64} for i in range(1,14)]
        with self.assertRaisesRegex(ValueError,'inventory'):manifest(saved)

    def test_whole_test_denominator_and_model_must_match_bounded_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);identity={'onnx_sha256':'o','checkpoint_sha256':'c'}
            metrics={'images':934,'denominators':{'observable_gt':2000},'dataset_audit_sha256':'d',
                     'slices':{'color/1':{}},'ap_by_iou':[{}]}
            saved={'identity':identity,'synthetic_test':metrics,
                   'paired_synthetic_test':{'ort':metrics,'simulator':metrics}}
            atomic_json(root/'exported/RESULT.json',{'onnx_sha256':'o'})
            atomic_json(root/'compiled/RESULT.json',{'identity':identity})
            atomic_json(root/'SYNTHETIC_TEST.json',{**metrics,'checkpoint_sha256':'c','scope':'HELD_OUT_SYNTHETIC_TEST'})
            paired={'identity':identity,'scope':'HELD_OUT_SYNTHETIC_TEST','images':934,
                    'backends':{'ort':metrics,'simulator':metrics}}
            atomic_json(root/'PAIRED_SYNTHETIC_TEST.json',paired);validate_reports(root,saved)
            bad=copy.deepcopy(paired);bad['backends']['simulator']['images']=12
            atomic_json(root/'PAIRED_SYNTHETIC_TEST.json',bad)
            with self.assertRaisesRegex(ValueError,'complete test denominator'):validate_reports(root,saved)

    def test_body_only_empty_gt_predictions_are_false_positives(self):
        prediction={'bbox_xyxy_visible':[0,100,50,200],'color_id':5}
        actual=compare_frame([prediction],[{'bbox_xyxy_visible':None,'visible_area_px':0,'color_id':5}])
        self.assertEqual((actual['tp'],actual['fp'],actual['fn']),(0,1,0))

    def test_wrong_color_is_separate_from_class_agnostic_localization(self):
        prediction={'bbox_xyxy_visible':[100,100,150,150],'color_id':5}
        target={'bbox_xyxy_visible':[100,100,150,150],'visible_area_px':2500,'color_id':1}
        actual=compare_frame([prediction],[target])
        self.assertEqual((actual['tp'],actual['fp'],actual['fn'],actual['wrong_color']),(1,0,0,1))

    def test_engineering_raw_fixture_cannot_be_final_error_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);atomic_json(root/'pred/RESULT.json',{'scope':'ENGINEERING_TRAIN_FIXTURE'})
            atomic_json(root/'data/AUDIT_V11.json',{'scenes':[]})
            with self.assertRaisesRegex(ValueError,'actual formal test'):
                render(root/'data',root/'pred',{},root/'unused')
            self.assertFalse((root/'unused').exists())

    def test_saved_raw_fixture_produces_bounded_miss_preview_without_any_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';pred=root/'pred';pred.mkdir();rows=[]
            for number in range(2):
                folder=data/f'scene-{number:05d}';folder.mkdir(parents=True)
                instances=[] if number==0 else [{'bbox_xyxy_visible':[200,140,260,200],
                    'visible_area_px':3600,'color_id':1}]
                atomic_json(folder/'labels.json',{'instances':instances})
                cv2.imwrite(str(folder/'rgb.png'),np.full((360,640,3),100,np.uint8))
                rows.append({'scene':folder.name,'split':'synthetic_test','rgb_sha256':sha(folder/'rgb.png'),
                             'label_sha256':sha(folder/'labels.json')})
            atomic_json(data/'AUDIT_V11.json',{'status':'PASS','scenes':rows})
            raw=np.zeros((2,1275,50),np.float32);raw[:,:,:6]=-20.
            files=[]
            for backend in ('ort','simulator'):
                path=pred/(backend+'-raw.npy');np.save(path,raw,allow_pickle=False)
                files.append({'name':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})
            identity={'model_wh':[320,192],'fixture_identity':'no model inference'}
            state={'status':'NNCASE_AND_ORT_TEST_RAW_COMPLETE','scope':'HELD_OUT_SYNTHETIC_TEST',
                'images':2,'completed_images':2,'dataset_index_sha256':sha(data/'AUDIT_V11.json'),
                'scene_order_sha256':order_sha(rows),'identity':identity,'files':files,'raw_shape':list(raw.shape)}
            atomic_json(pred/'RESULT.json',state)
            evidence=render(data,pred,{'identity':identity,'scope':'HELD_OUT_SYNTHETIC_TEST'},root/'previews')
            self.assertEqual(evidence['new_inference_passes'],0)
            self.assertEqual(evidence['error_images_at_registered_rules'],1)
            self.assertEqual(len(evidence['examples']),1)
            self.assertEqual(evidence['examples'][0]['measurements']['simulator']['fn'],1)
            self.assertTrue((root/'previews/case-01.jpg').is_file())


if __name__=='__main__':unittest.main()
