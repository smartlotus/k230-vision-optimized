"""Host serialization/lineage fixtures only, never evidence of actual KPU execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parent/'board'))
from topface_pair_capture import PairCapture,write_planar,write_raw
from evaluate_board_pairs import load_pair,compare_native_decoded


class BoardPairTests(unittest.TestCase):
    def identity(self):
        return {'model_wh':[320,192],'output_shape':[1,1275,50],'input':'NCHW_RGB_UINT8',
                'kmodel_sha256':'a'*64,'onnx_sha256':'b'*64}

    def fixture(self,root):
        (root/'output_contract.json').write_text('{"fixture":"no actual board"}',encoding='utf8')
        source=np.zeros((3,360,640),np.uint8);source[0]=7;source[1]=91;source[2]=203
        prepared=np.zeros((1,3,192,320),np.uint8);prepared[:,0]=3;prepared[:,1]=33;prepared[:,2]=133
        raw=np.arange(1275*50,dtype=np.float32).reshape(1,1275,50)/1000
        capture=PairCapture(str(root),self.identity(),'HOST_FIXTURE_ONLY',1)
        pending=capture.start(source,prepared,5,1234);capture.finish(pending,raw,{'detections':[]})
        result=capture.result();return result,source,prepared,raw

    def test_planar_rgb_and_little_endian_raw_round_trip_with_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,source,prepared,raw=self.fixture(root)
            self.assertEqual(result['status'],'PAIRED_CAPTURE_COMPLETE')
            self.assertFalse(result['performance_measurement'])
            contract=hashlib.sha256((root/'output_contract.json').read_bytes()).hexdigest()
            values=load_pair(root/'pair-01',result['samples'][0],self.identity(),contract)
            np.testing.assert_array_equal(values['source'][0],source)
            np.testing.assert_array_equal(values['prepared'],prepared)
            np.testing.assert_array_equal(values['raw'],raw)
            self.assertEqual((root/'pair-01/source.rgb888p').stat().st_size,691200)
            self.assertEqual((root/'pair-01/prepared.rgb888p').stat().st_size,184320)
            self.assertEqual((root/'pair-01/kpu-raw.f32le').stat().st_size,255000)

    def test_changed_native_raw_or_wrong_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,_,_,_=self.fixture(root);record=result['samples'][0]
            contract=hashlib.sha256((root/'output_contract.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'identity'):
                load_pair(root/'pair-01',record,{**self.identity(),'kmodel_sha256':'c'*64},contract)
            (root/'pair-01/kpu-raw.f32le').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'byte count/hash'):
                load_pair(root/'pair-01',record,self.identity(),contract)

    def test_existing_pair_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,source,prepared,_=self.fixture(root)
            original=(root/'PAIR_CAPTURE_RESULT.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'Preserve existing'):
                PairCapture(str(root),self.identity(),'HOST_FIXTURE_ONLY',1)
            self.assertEqual((root/'PAIR_CAPTURE_RESULT.json').read_bytes(),original)

    def test_partial_input_capture_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'output_contract.json').write_text('{}')
            capture=PairCapture(str(root),self.identity(),'HOST_FIXTURE_ONLY',1)
            result=capture.result();self.assertEqual(result['status'],'PAIRED_CAPTURE_INCOMPLETE')

    def test_nonplanar_or_fractional_source_cannot_be_written_as_uint8_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'bad.rgb')
            with self.assertRaisesRegex(ValueError,'shape'):
                write_planar(path,np.zeros((2,2,3),np.uint8),2,2)
            with self.assertRaisesRegex(ValueError,'uint8-valued'):
                write_planar(path,np.full((3,2,2),.5,np.float32),2,2)
            with self.assertRaisesRegex(ValueError,'shape'):
                write_raw(path,np.zeros((1,5,50),np.float32),1275)

    def test_captured_postprocess_is_compared_including_geometry_flags_and_capacity(self):
        from task_geometry import decode
        raw=np.zeros((1275,50),np.float32);raw[:,:6]=-20.;raw[200,0]=5.;raw[200,44:46]=[.3,.7]
        captured=decode(raw)
        self.assertTrue(compare_native_decoded(raw,captured,self.identity())['pass'])
        captured['detections'][0]['center_visible_probability']=.123
        captured['detections'][0]['ellipse']['orientation_valid']=not captured['detections'][0]['ellipse']['orientation_valid']
        captured['capacity_state']='INVALID_FIXTURE'
        compared=compare_native_decoded(raw,captured,self.identity())
        self.assertFalse(compared['pass']);self.assertGreaterEqual(compared['difference_count'],3)


if __name__=='__main__':unittest.main()
