import json
import tempfile
import unittest
from pathlib import Path
from k230.visual_inspector import VisualInspector
from vision.top import filter_surfaces
from test_block_detector import Image,Blob
from test_block_service import frame


PROFILE={'width':320,'height':240,'crop':False,'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}


def top_profile():
    return {'schema_version':1,'provenance':{'source':'HOST_TEST_FIXTURE_NOT_HARDWARE'},
        'TOP_CONFIG':{'BUFFER':{'camera_id':2,'image_profile':dict(PROFILE),
            'thresholds':{'1':[5,95,24,65,7,55]},'roi':[0,0,320,240],'reference':[160,120],
            'min_area':300,'max_area':12000,
            'surface_shape':{'min_axis_ratio':0.75,'max_axis_ratio':1.35,'min_density':0.60}}}}


class InspectorTests(unittest.TestCase):
    def test_modes_bind_cameras_and_reject_old_or_invalid_commands(self):
        inspector=VisualInspector()
        self.assertEqual(inspector.camera_id(),2)
        self.assertTrue(inspector.apply({'seq':1,'mode':'QR','target_id':None,'station':'QR'}))
        self.assertEqual(inspector.camera_id(),1)
        self.assertFalse(inspector.apply({'seq':1,'mode':'COLOR','target_id':1,'station':'RAW'}))
        self.assertEqual(inspector.mode,'QR')
        for cmd in ({'seq':2,'mode':'TOP','target_id':True,'station':'BUFFER'},
                    {'seq':2,'mode':'RING','target_id':6,'station':'PROCESS'},
                    {'seq':2,'mode':'COLOR','target_id':1,'station':'RAW','motion_enabled':True}):
            self.assertFalse(inspector.apply(cmd))
        self.assertEqual(inspector.seq,1)

    def test_unsampled_top_never_reuses_the_side_color_profile(self):
        inspector=VisualInspector('TOP',1,'BUFFER')
        result=inspector.process(frame(Image([Blob(1,(100,70,40,40),1200)])))
        self.assertEqual(result['reason'],'TOP_PARAMETERS_REQUIRED')
        self.assertIsNone(result['u_px'])

    def test_bound_top_face_is_pixel_only_and_rejects_tall_body(self):
        inspector=VisualInspector('TOP',1,'BUFFER',top_profile())
        image=Image([Blob(1,(100,70,40,40),1200)])
        for i in range(1,4):result=inspector.process(frame(image,i))
        self.assertEqual((result['u_px'],result['v_px']),(120,90))
        self.assertTrue(result['stable'])
        self.assertEqual(result['status'],'UNCALIBRATED')
        self.assertIsNone(result['x_mm'])
        result=inspector.process(frame(Image([Blob(1,(100,70,25,70),1400)]),4))
        self.assertEqual(result['status'],'NOT_FOUND')
        self.assertIsNone(result['u_px'])
        changed=frame(image,5);changed['image_profile']=dict(PROFILE,pose_id='different')
        self.assertEqual(inspector.process(changed)['reason'],'TOP_PROFILE_BINDING_MISMATCH')

    def test_pixel_profile_cannot_load_metric_or_actuator_settings(self):
        inspector=VisualInspector()
        with self.assertRaises(ValueError):inspector.load_profile({'schema_version':1,'CALIBRATIONS':{'fake':True}})
        self.assertEqual(inspector.config['CALIBRATIONS'],{})
        self.assertFalse(inspector.status()['motors_enabled'])
        bad=top_profile();bad['TOP_CONFIG']['BUFFER']['block_parameters']={'thresholds':{1:[]}}
        with self.assertRaises(ValueError):inspector.load_profile(bad)

    def test_file_controls_are_bounded_and_bad_json_keeps_current_mode(self):
        inspector=VisualInspector()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control.json'
            path.write_text('{bad',encoding='utf-8')
            self.assertFalse(inspector.poll_file(path))
            self.assertEqual(inspector.mode,'COLOR')
            path.write_text(json.dumps({'seq':1,'mode':'IDLE','target_id':None,'station':None}),encoding='utf-8')
            self.assertTrue(inspector.poll_file(path))
            self.assertIsNone(inspector.process(None))
            path.write_text('x'*1025,encoding='utf-8')
            self.assertFalse(inspector.poll_file(path))

    def test_invalid_frame_clears_cached_color_candidates(self):
        inspector=VisualInspector()
        inspector.process(frame(Image([Blob(1,(50,60,30,60),1400)])))
        self.assertTrue(inspector.adapter.last_block_candidates)
        inspector.capture_failed()
        self.assertEqual(inspector.adapter.last_block_candidates,[])
        inspector.process(None)
        self.assertEqual(inspector.adapter.last_block_candidates,[])
        inspector.suspend('CSI2_UNAVAILABLE')
        self.assertEqual(inspector.mode,'IDLE')
        self.assertEqual(inspector.status()['control_error'],'CSI2_UNAVAILABLE')

    def test_surface_limits_are_required_and_do_not_guess_a_profile(self):
        with self.assertRaises(ValueError):filter_surfaces([],None)
        with self.assertRaises(ValueError):filter_surfaces([],{'min_axis_ratio':1,'max_axis_ratio':0,'min_density':.6})

    def test_measured_black_surface_override_does_not_relax_other_colors(self):
        shape={'min_axis_ratio':.65,'max_axis_ratio':1.4,'min_density':.55,'per_color':{'5':{'min_density':.45}}}
        candidates=[{'target_id':c,'bbox':(10,10,20,24),'pixels':240} for c in (1,5)]
        self.assertEqual([c['target_id'] for c in filter_surfaces(candidates,shape)],[5])
        shape['per_color']['5']['min_density']=-1
        with self.assertRaises(ValueError):filter_surfaces(candidates,shape)

    def test_switching_modes_clears_color_tracks_and_requires_fresh_confirmation(self):
        inspector=VisualInspector()
        image=Image([Blob(1,(50,60,30,60),1400)])
        for i in range(1,4):inspector.process(frame(image,i))
        self.assertTrue(inspector.adapter.last_block_candidates[0]['stable'])
        inspector.apply({'seq':1,'mode':'TOP','target_id':1,'station':'BUFFER'})
        self.assertEqual(inspector.adapter.last_block_candidates,[])
        inspector.apply({'seq':2,'mode':'COLOR','target_id':1,'station':'RAW'})
        inspector.process(frame(image,4))
        self.assertFalse(inspector.adapter.last_block_candidates[0]['stable'])

    def test_qr_scan_reopens_after_timeout_without_clearing_session_deduplication(self):
        inspector=VisualInspector('QR',None,'QR')
        def detect(*args):
            inspector.adapter.qr.started_ms=123
            inspector.adapter.qr.submitted.add('156+123+516+231')
            return {'status':'NOT_FOUND','reason':'QR_TIMEOUT'}
        inspector.adapter.detect=detect
        self.assertEqual(inspector.process(None)['reason'],'QR_TIMEOUT')
        self.assertIsNone(inspector.adapter.qr.started_ms)
        self.assertIn('156+123+516+231',inspector.adapter.qr.submitted)


if __name__=='__main__':unittest.main()
