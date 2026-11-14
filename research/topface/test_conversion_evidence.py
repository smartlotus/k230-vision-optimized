"""Reject substituted saved runs and retain quantization geometry losses."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_store import atomic_json, sha
from compile_trained_topface import compare_detections
from publisher.plan_saved_output import plan


class ConversionEvidenceTests(unittest.TestCase):
    def test_original_save_response_binds_provider_normalized_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            notebook=root/'source.ipynb';notebook.write_text('{}',encoding='utf8')
            atomic_json(root/'SUBMISSION_INTENT.json',{'kernel':'owner/requested-slug','run_id':'k230-fixture-a01',
                'input_manifest_sha256':'a'*64,'notebook_file':notebook.name,'notebook_sha256':sha(notebook)})
            atomic_json(root/'KERNEL_SAVE_RESPONSE.json',{'ref':'/code/owner/normalized-slug','kernelId':123,'versionNumber':1})
            observed=root/'observed.json'
            provider={'status':'COMPLETE','ref':'owner/normalized-slug','kernel_id':123,'version':1}
            atomic_json(observed,provider)
            result=root/'result.json'
            atomic_json(result,{'run_id':'k230-fixture-a01','input_manifest_sha256':'a'*64,
                'archive':{'name':'k230-fixture-a01.tar.gz','bytes':100,'sha256':'b'*64}})
            self.assertEqual(plan(root,observed,result)['prep']['ref'],'owner/normalized-slug')
            atomic_json(observed,{**provider,'kernel_id':124})
            with self.assertRaisesRegex(ValueError,'original Save response'):
                plan(root,observed,result)

    def test_perfect_box_does_not_hide_lost_ellipse_and_angle(self):
        detection={'bbox_xyxy_visible':[0,0,10,10],'color_id':1,'score':.9,
                   'center_xy_projected_3d':[5,5],
                   'ellipse':{'orientation_valid':True,'conic_center_xy':[5,5],
                              'semi_major':5,'semi_minor':3,'orientation_deg':0}}
        expected={'detections':[detection],'capacity_state':'OK'}
        actual={'detections':[{**detection,'ellipse':None}],'capacity_state':'OK'}
        with patch('compile_trained_topface.decode',side_effect=[expected,actual]):
            result=compare_detections(None,None,320,192)
        self.assertEqual(result['matched_count'],1)
        self.assertEqual(result['geometry_transitions']['finite_to_missing'],1)
        self.assertEqual(result['geometry_transitions']['angle_valid_to_invalid'],1)
        self.assertFalse(result['matches'][0]['simulator_ellipse_finite'])


if __name__=='__main__':unittest.main()
