"""Controlled fixture checks; no real-camera acceptance is claimed by these tests."""
import unittest
import cv2
import numpy as np
from tools.evaluate_rough_ring_alignment import evaluate


class RoughAlignmentReplayTests(unittest.TestCase):
    def fixture(self):
        width,height=320,240
        bgr=np.full((height,width,3),190,dtype=np.uint8)
        for radius in (38,60):cv2.circle(bgr,(160,120),radius,(100,100,100),2)
        cv2.circle(bgr,(160,115),24,(60,60,90),-1)
        native=cv2.cvtColor(bgr,cv2.COLOR_BGR2BGR565).tobytes()
        yy,xx=np.indices((60,80));x,y=2+xx*4,2+yy*4
        grid=np.zeros((60,80,3),dtype=np.int16);grid[:,:,0]=70
        grid[(x-160)**2+(y-115)**2<24**2]=[30,40,30]
        sample=dict(source='REAL_CAMERA_NATIVE_LAB',is_simulated=False,camera_id=2,mode='RING',
            width=width,height=height,stride=4,x0=2,y0=2,lab_grid=grid.tolist(),
            frame_id=1,timestamp_ms=1000,image_profile={'width':width,'height':height,'pose_id':None})
        return sample,native

    def test_independent_paper_edges_and_material_center_give_raw_pixel_offset(self):
        sample,native=self.fixture()
        r=evaluate(sample,native,[8,8,304,224],[148,103,24,24])
        self.assertEqual(len(r['material_candidates']),1)
        self.assertEqual(len(r['paper_candidates']),1)
        self.assertAlmostEqual(r['alignment']['dx_px'],0,delta=2)
        self.assertAlmostEqual(r['alignment']['dy_px'],-5,delta=2)
        self.assertEqual(r['alignment']['reason'],'STABILITY_REQUIRED')
        self.assertIsNone(r['alignment']['quality'])
        self.assertIsNone(r['official_points'])

    def test_wrong_source_frame_binding_or_byte_count_rejects_before_analysis(self):
        sample,native=self.fixture()
        for fields in ({'camera_id':1},{'is_simulated':True},{'frame_id':None},
                       {'image_profile':{'width':640,'height':360}}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                evaluate(dict(sample,**fields),native,[8,8,304,224],[148,103,24,24])
        with self.assertRaises(ValueError):evaluate(sample,native[:-2],[8,8,304,224],[148,103,24,24])

    def test_interior_roi_outside_observation_is_not_silently_used(self):
        sample,native=self.fixture()
        with self.assertRaises(ValueError):evaluate(sample,native,[100,80,160,140],[20,20,24,24])


if __name__=='__main__':unittest.main()
