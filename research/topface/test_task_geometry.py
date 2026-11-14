import unittest
import numpy as np
from task_geometry import letterbox_parameters,priors,decode


class TaskGeometryTests(unittest.TestCase):
    def test_letterbox_and_inverse_are_exact_and_grid_has_no_half_stride(self):
        for wh,resize,padding,gain,n in (((224,128),(224,126),(1,1,0,0),.35,596),
              ((320,192),(320,180),(6,6,0,0),.5,1275),((384,224),(384,216),(4,4,0,0),.6,1788)):
            info=letterbox_parameters(*wh);self.assertEqual(info['resized_wh'],list(resize))
            self.assertEqual(info['padding_tblr'],list(padding));self.assertEqual(info['gain'],gain)
            xy=np.array([233.25,175.5]);offset=np.array([padding[2],padding[0]])
            np.testing.assert_allclose((xy*gain+offset-offset)/gain,xy,atol=1e-10)
            point,scale=priors(*wh);self.assertEqual(len(point),n);np.testing.assert_equal(point[0],[0,0])
            np.testing.assert_equal(point[1],[8,0]);self.assertEqual(scale[0],8)

    def test_five_detections_and_distinct_physical_conic_centers(self):
        point,scale=priors(320,192);raw=np.zeros((len(point),50),np.float32);raw[:,:6]=-20
        raw[:,6:38]=-20;raw[:,6:38].reshape(-1,4,8)[:,:,1]=20
        ids=[3*40+4,3*40+12,3*40+20,3*40+28,12*40+4]
        for color,index in enumerate(ids):
            raw[index,color]=5;raw[index,38:40]=[.25,.5];raw[index,48:50]=[.5,.75]
            raw[index,42:44]=[0,-.4];raw[index,44:46]=[0,1]
        result=decode(raw);self.assertEqual(result['count'],5)
        self.assertEqual(result['capacity_state'],'OVER_CAPACITY')
        for det in result['detections']:
            self.assertGreater(np.linalg.norm(np.array(det['center_xy_projected_3d'])-det['ellipse']['conic_center_xy']),0)
            self.assertGreater(det['score'],.99)

    def test_nonfinite_or_wrong_contract_is_rejected(self):
        with self.assertRaises(ValueError):decode(np.zeros((1275,48),np.float32))
        raw=np.zeros((1275,50),np.float32);raw[0,0]=np.nan
        with self.assertRaises(ValueError):decode(raw)

    def test_padding_false_positives_do_not_consume_five_valid_slots(self):
        raw=np.zeros((1275,50),np.float32);raw[:,:6]=-20
        raw[:,6:38]=-20;raw[:,6:38].reshape(-1,4,8)[:,:,1]=20
        valid=[3*40+4,3*40+12,3*40+20,3*40+28,12*40+4]
        for index in valid:raw[index,0]=5
        for index in (4,12,20):
            raw[index,0]=10;raw[index,6:38]=-20
            raw[index,6:38].reshape(4,8)[:,0]=20  # degenerate box wholly in top padding
        result=decode(raw)
        self.assertEqual(result['count'],5);self.assertEqual(result['capacity_state'],'OVER_CAPACITY')


if __name__=='__main__':unittest.main()
