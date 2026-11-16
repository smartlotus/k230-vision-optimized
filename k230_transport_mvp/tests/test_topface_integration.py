import copy
import unittest
from k230.vision_adapter import VisionAdapter
from k230.topface_service import configuration
from k230.topface_adapter import MODEL_SHA


class Image:
    def width(self):return 640
    def height(self):return 360


class Backend:
    identity={'kmodel_sha256':MODEL_SHA}
    def __init__(self):
        self.closed=False
        self.result={'capacity_state':'OK','detections':[{
            'color_id':1,'boundary_candidate':False,'center_valid_probability':1.,
            'center_visible_probability':1.,'bbox_xyxy_visible':[100,100,140,140],
            'center_xy_projected_3d':[118,121],'score':.9,
            'ellipse':{'geometry_confidence':.99}}]}
    def infer(self,image):return self.result
    def close(self):self.closed=True


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=configuration('/sdcard/model')
        self.backend=Backend();self.adapter=VisionAdapter(self.cfg,topface_backend=self.backend)
        self.frame={'image':Image(),'camera_id':2,'frame_id':1,'timestamp_ms':100,'is_simulated':False,
                    'image_profile':{'width':640,'height':360,'crop':False,'hmirror':False,
                                     'vflip':False,'rotation':0,'pose_id':None}}
    def detect(self):return self.adapter.detect('TOP',self.frame,1,'BUFFER')

    def test_native_pixels_still_require_measured_calibration(self):
        out=self.detect()
        self.assertEqual(out['status'],'UNCALIBRATED')
        self.assertEqual((out['u_px'],out['v_px']),(118,121))
        self.assertIsNone(out['x_mm']);self.assertFalse(out['target_pose_valid'])
        self.assertEqual(out['model_confidence'],.9)
        self.assertEqual(out['model_sha256'],MODEL_SHA)

    def test_capacity_boundary_hidden_center_and_geometry_never_return_coordinates(self):
        for state in ('OVER_CAPACITY','UNCERTAIN'):
            self.backend.result['capacity_state']=state
            out=self.detect();self.assertEqual(out['status'],'UNAVAILABLE')
            self.assertEqual(out['capacity_state'],state);self.assertIsNone(out['u_px'])
        self.backend.result['capacity_state']='OK'
        for key,value in [('boundary_candidate',True),('center_visible_probability',.1),
                          ('center_valid_probability',.1),('ellipse',None)]:
            original=copy.deepcopy(self.backend.result['detections'][0])
            self.backend.result['detections'][0][key]=value
            out=self.detect();self.assertEqual(out['status'],'NOT_FOUND');self.assertIsNone(out['u_px'])
            self.backend.result['detections'][0]=original

    def test_consumed_exclusion_and_camera_source_gate(self):
        self.adapter.mark_consumed('BUFFER',1,118,121,10)
        self.assertEqual(self.detect()['status'],'NOT_FOUND')
        self.frame['camera_id']=1
        self.assertEqual(self.detect()['reason'],'FRAME_SOURCE_MISMATCH')

    def test_close_model_and_legacy_default(self):
        self.adapter.close();self.assertTrue(self.backend.closed)
        self.assertIsNone(VisionAdapter(self.cfg).topface_backend)

    def test_configuration_keeps_roles_and_rejects_uart_or_transformed_input(self):
        self.assertEqual(self.cfg['MODE_CAMERA']['TOP'],2)
        self.assertEqual(self.cfg['MODE_CAMERA']['QR'],1)
        for overrides in ({'UART_ENABLED':True},{'CROP':True},{'WIDTH':320},{'HMIRROR':True}):
            with self.assertRaises(ValueError):configuration('/sdcard/model',overrides)


if __name__=='__main__':unittest.main()
