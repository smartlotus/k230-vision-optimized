import copy
import unittest
import numpy as np
from tools.kpu_pipeline_probe import latency,timing_proposals,probe,sha_file,scalar_timing_proposals
from tools.check_canmv_connection import analyze,live_readiness
from vision.yolo_candidates import input_contract
from unittest.mock import patch,Mock
import tempfile
from pathlib import Path
import hashlib


class PipelineProbeTests(unittest.TestCase):
    def test_scalar_postprocess_matches_positive_geometry_without_numpy_work_buffers(self):
        raw=np.zeros((1,5,1029),np.float32);raw[0,4,2]=-.00035
        raw[0,:,0]=[112,112,28,14,.9];raw[0,:,1]=[112,112,28,14,.8]
        result,validation=scalar_timing_proposals(raw,input_contract(224))
        self.assertEqual(len(result['proposals']),1)
        np.testing.assert_allclose(result['proposals'][0]['bbox_xywh'],[280,160,80,40])
        self.assertFalse(validation['raw_probability_range_valid'])
    def test_stale_ready_flag_requires_actual_echo_and_idle(self):
        client=Mock();client.tool.side_effect=[{'connected':True,'ready':False},
            {'running':False},{'status':'ok'},{'text':"print('READINESS_1112')\r\nREADINESS_1112\r\n>>> "}]
        result=live_readiness(client,'READINESS_1112')
        self.assertFalse(result['cached_ready']);self.assertTrue(result['fresh_repl_echo'])
        client=Mock();client.tool.side_effect=[{'connected':True,'ready':True},{'running':True}]
        with self.assertRaises(RuntimeError):live_readiness(client,'READINESS_1112')
        self.assertFalse(any(c.args[0]=='canmv_terminal_input' for c in client.tool.call_args_list))
    def test_chunked_hash_yields_and_preserves_all_input_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.bin';raw=bytes(range(256))*33;path.write_bytes(raw)
            with patch('tools.kpu_pipeline_probe.cooperate') as cooperate:
                actual=sha_file(path)
            self.assertEqual(actual,hashlib.sha256(raw).hexdigest())
            self.assertEqual(cooperate.call_count,3)
    def test_no_native_access_before_explicit_bounds(self):
        with self.assertRaises(ValueError):probe('/absent',iterations=100)
        with self.assertRaises(ValueError):probe('/absent',image_limit=9)
        self.assertAlmostEqual(latency([10,20])['fps'],1000/15)

    def test_quantized_projection_retains_raw_failure_and_input_bytes(self):
        raw=np.zeros((1,5,1029),np.float32)
        raw[0,4,0]=-.00035
        before=raw.copy()
        proposals,validation=timing_proposals(raw,input_contract(224),np)
        self.assertTrue(np.array_equal(raw,before))
        self.assertFalse(validation['raw_probability_range_valid'])
        self.assertEqual(validation['scores_projected'],1)
        self.assertFalse(validation['task_accuracy_verified'])
        self.assertEqual(proposals['proposals'],[])

    def test_scalar_nms_preserves_source_transform_and_capacity(self):
        raw=np.zeros((1,5,1029),np.float32)
        raw[0,:,0]=[112,112,28,14,.9];raw[0,:,1]=[112,112,28,14,.8]
        result,_=timing_proposals(raw,input_contract(224),np)
        self.assertEqual(len(result['proposals']),1)
        np.testing.assert_allclose(result['proposals'][0]['bbox_xywh'],[280,160,80,40])
        raw[0,0:4,:65]=np.array([112,112,28,14],np.float32)[:,None]
        raw[0,4,:65]=.9
        result,_=timing_proposals(raw,input_contract(224),np)
        self.assertFalse(result['complete']);self.assertEqual(result['proposals'],[])

    def test_large_overshoot_never_reaches_native_decoder(self):
        for score in (-.01,1.01,float('nan')):
            raw=np.zeros((1,5,1029),np.float32);raw[0,4,0]=score
            with patch('vision.yolo_candidates.decode_material_proposals') as decoder:
                with self.assertRaises(ValueError):timing_proposals(raw,input_contract(224),np)
                decoder.assert_not_called()

    def test_log_correlation_is_not_hardware_cause(self):
        report=analyze('No frames received for startup; restarting preview\nboard disconnected source=poller\n')
        self.assertEqual(report['preview_no_frame_restarts'],1)
        self.assertTrue(report['preview_restarts_before_disconnect'])
        self.assertIn('correlation',report['inference'])
