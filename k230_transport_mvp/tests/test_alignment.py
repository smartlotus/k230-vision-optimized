import unittest
from vision.alignment import measure_alignment,evaluate_placement


class AlignmentTests(unittest.TestCase):
    def observation(self, **fields):
        value=dict(status='GEOMETRY_ONLY',camera_id=2,frame_id=10,timestamp_ms=1000,is_simulated=True,
            image_profile={'width':640,'height':360,'pose_id':'synthetic'},
            u_px=100,v_px=100,stable=True)
        value.update(fields);return value

    def test_signed_xy_error_and_pixel_quality_have_no_mm_or_score_authority(self):
        r=measure_alignment(self.observation(u_px=103,v_px=104),self.observation(),now_ms=1100)
        self.assertEqual((r['dx_px'],r['dy_px'],r['error_px'],r['quality']),(3,4,5,50))
        self.assertIsNone(r['x_mm']);self.assertIsNone(r['official_points'])
        self.assertFalse(r['target_pose_valid'])

    def test_different_frame_camera_binding_or_source_cannot_be_combined(self):
        for fields in ({'frame_id':11},{'camera_id':1},{'is_simulated':False},{'timestamp_ms':1001},
                       {'image_profile':{'width':640,'height':360,'pose_id':'moved'}}):
            r=measure_alignment(self.observation(),self.observation(**fields),now_ms=1100)
            self.assertEqual(r['reason'],'SAME_FRAME_SOURCE_REQUIRED')
            self.assertIsNone(r['error_px'])

    def test_stale_future_and_bad_centers_stay_unknown(self):
        for fields in ({'timestamp_ms':500},{'timestamp_ms':1200},{'u_px':float('nan')},
                       {'v_px':360},{'u_px':True}):
            r=measure_alignment(self.observation(**fields),self.observation(**fields),now_ms=1100)
            self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['quality'])

    def test_unstable_has_raw_pixel_error_but_no_quality(self):
        r=measure_alignment(self.observation(u_px=105,stable=False),self.observation(),now_ms=1100)
        self.assertEqual(r['error_px'],5);self.assertIsNone(r['quality'])

    def test_alignment_does_not_infer_placement_completion_or_scoring_zone(self):
        r=evaluate_placement(self.observation(),self.observation(),3,None,now_ms=1100)
        self.assertEqual(r['alignment']['quality'],100)
        self.assertIsNone(r['placement_score']['official_points'])
        self.assertFalse(r['motion_authorized'])

    def test_verified_receipt_uses_slot_and_zone_independently_of_pixel_quality(self):
        receipt=dict(status='DONE',action='PLACE',placement_verified=True,
            score_observation_verified=True,is_simulated=True,observed_slot=3,
            scoring_zone=5,placement_state='IN_RING',stable=True,tilted=False)
        r=evaluate_placement(self.observation(),self.observation(),3,receipt,
            now_ms=1100,calibrated=True)
        self.assertEqual(r['alignment']['quality'],100)
        self.assertEqual(r['placement_score']['official_points'],3)


if __name__=='__main__':unittest.main()
