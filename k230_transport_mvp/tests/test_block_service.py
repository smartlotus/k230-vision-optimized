import copy
import unittest
from k230.block_profile import make_config,PARAMETERS
from k230.vision_adapter import VisionAdapter
from ports.protocol import encode_observation,StreamParser
from test_block_detector import Image,Blob


def frame(image,number=1):
    return {'image':image,'camera_id':2,'frame_id':number,'timestamp_ms':number*40,
            'is_simulated':False,'width':320,'height':240,
            'image_profile':{'width':320,'height':240,'crop':False,'hmirror':False,
                'vflip':False,'rotation':0,'pose_id':None}}


class BlockServiceTests(unittest.TestCase):
    def test_dual_camera_binding_and_no_unmeasured_metric_output(self):
        config=make_config()
        self.assertEqual(config['MODE_CAMERA']['QR'],1)
        self.assertEqual(config['MODE_CAMERA']['COLOR'],2)
        self.assertEqual(config['CAMERA_IDS'],(2,1))
        adapter=VisionAdapter(config)
        image=Image([Blob(1,(20,60,30,50),1100)])
        for i in range(1,4):result=adapter.detect('COLOR',frame(image,i),target_id=1,station='RAW')
        self.assertEqual(result['status'],'UNCALIBRATED')
        self.assertEqual(result['coord_frame'],'IMAGE_PX')
        self.assertEqual(result['u_px'],35)
        self.assertTrue(result['stable'])
        self.assertFalse(result['is_simulated'])
        self.assertIsNone(result['x_mm'])
        self.assertFalse(result['target_pose_valid'])
        self.assertTrue(result['color_profile_sampling_verified'])
        self.assertEqual(result['color_profile_measurement_camera_id'],2)
        parsed=StreamParser().feed(encode_observation(1,result))[0]
        self.assertEqual(parsed['camera_id'],2)
        self.assertEqual(parsed['status'],'UNCALIBRATED')

    def test_swapped_service_rejects_old_camera_as_current_object_source(self):
        adapter=VisionAdapter(make_config())
        wrong=frame(Image([Blob(1,(20,60,30,50),1100)]))
        wrong['camera_id']=1
        result=adapter.detect('COLOR',wrong,1,'RAW')
        self.assertEqual(result['reason'],'FRAME_SOURCE_MISMATCH')
        self.assertIsNone(result['u_px'])
        self.assertIsNone(result['x_mm'])

    def test_clipped_consumed_and_missing_targets_never_reuse_previous_center(self):
        adapter=VisionAdapter(make_config())
        image=Image([Blob(1,(20,60,30,50),1100)])
        result=adapter.detect('COLOR',frame(image),target_id=1,station='RAW')
        adapter.mark_consumed('RAW',1,result['u_px'],result['v_px'],10)
        result=adapter.detect('COLOR',frame(image,2),target_id=1,station='RAW')
        self.assertEqual(result['status'],'NOT_FOUND')
        self.assertIsNone(result['u_px'])
        adapter.clear_consumed('RAW')
        result=adapter.detect('COLOR',frame(Image([Blob(1,(0,30,30,50),1100)]),3),target_id=1,station='RAW')
        self.assertEqual(result['status'],'NOT_FOUND')

    def test_unknown_color_and_top_profile_are_explicit(self):
        config=copy.deepcopy(make_config())
        config['COLOR_CONFIG']['RAW']['block_parameters']['thresholds'][5]=None
        adapter=VisionAdapter(config)
        result=adapter.detect('COLOR',frame(Image()),target_id=5,station='RAW')
        self.assertEqual(result['reason'],'COLOR_SAMPLE_REQUIRED')
        result=adapter.detect('TOP',frame(Image()),target_id=1,station='BUFFER')
        self.assertEqual(result['reason'],'TOP_PARAMETERS_REQUIRED')


if __name__=='__main__':unittest.main()
