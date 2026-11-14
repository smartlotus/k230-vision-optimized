import unittest
import numpy as np
from board.topface_decode import decode as portable
from task_geometry import decode as host,priors


class PortableDecoderTests(unittest.TestCase):
    def compare(self,raw,width,height):
        expected=host(raw,width,height)
        actual=portable(lambda i,c:raw[i,c],width,height,len(raw),50)
        for key in ('count','post_nms_total_before_five_limit','capacity_state','uncertain'):
            self.assertEqual(expected[key],actual[key])
        for a,b in zip(expected['detections'],actual['detections']):
            self.assertEqual(a['color_id'],b['color_id']);self.assertAlmostEqual(a['score'],b['score'],places=6)
            for key in ('bbox_xyxy_visible','center_xy_projected_3d'):
                np.testing.assert_allclose(a[key],b[key],atol=1e-4)
            if a['ellipse'] is None:self.assertIsNone(b['ellipse'])
            else:
                for key in ('semi_major','semi_minor','orientation_deg'):
                    self.assertAlmostEqual(a['ellipse'][key],b['ellipse'][key],places=4)
                np.testing.assert_allclose(a['ellipse']['conic_center_xy'],b['ellipse']['conic_center_xy'],atol=1e-4)

    def test_export_shapes_seeded_five_object_and_empty_cases_match_host(self):
        rng=np.random.default_rng(20261114)
        for width,height in ((224,128),(320,192),(384,224)):
            point,_=priors(width,height);raw=rng.normal(0,.3,(len(point),50)).astype(np.float32)
            raw[:,:6]=-20;raw[:,6:38]=-20;raw[:,6:38].reshape(-1,4,8)[:,:,1]=20
            self.compare(raw,width,height)
            for color,xy in enumerate(((32,32),(80,32),(128,32),(176,32),(32,96))):
                index=int(np.argmin(np.linalg.norm(point-np.array(xy),axis=1)))
                raw[index,color]=5;raw[index,44:46]=[.6,.8]
            self.compare(raw,width,height)

    def test_portable_contract_rejects_wrong_shape_and_nonfinite_logit(self):
        with self.assertRaises(ValueError):portable(lambda i,c:0,positions=596)
        raw=np.zeros((1275,50));raw[0,0]=float('nan')
        with self.assertRaises(ValueError):portable(lambda i,c:raw[i,c])


if __name__=='__main__':unittest.main()
