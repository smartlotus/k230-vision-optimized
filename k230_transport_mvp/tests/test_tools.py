import copy
import tempfile
from pathlib import Path
import unittest

from tools.calibrate_homography import calibrate,fit_homography,project


def synthetic_data():
    return {'calibration_id':'SYNTHETIC_ONLY','camera_id':2,'plane':'RAW_TOP',
            'station':'RAW','pose_id':'SIM_POSE','image_profile':{'width':640,'height':480,'pose_id':'SIM_POSE',
            'crop':False,'hmirror':False,'vflip':False,'rotation':0},
            'is_simulated':True,
            'training':[{'px':p,'mm':[p[0]*2+10,p[1]*3-4]} for p in [[0,0],[100,0],[100,100],[0,100],[30,45],[75,15]]],
            'validation':[{'px':p,'mm':[p[0]*2+10,p[1]*3-4]} for p in [[50,50],[25,75]]]}


class CalibrationToolTests(unittest.TestCase):
    def test_independent_points_validate_synthetic_transform(self):
        result=calibrate(synthetic_data(),0.001)
        self.assertLess(result['max_error_mm'],0.001)
        self.assertTrue(result['is_simulated'])
        self.assertFalse(result['target_pose_valid'])

    def test_missing_real_measurements_do_not_export(self):
        data=synthetic_data()
        data['training'][0]['mm']=None
        with self.assertRaises((ValueError,TypeError)):
            calibrate(data,1)

    def test_validation_reuse_is_rejected(self):
        data=synthetic_data()
        data['validation'][0]=copy.deepcopy(data['training'][0])
        with self.assertRaises(ValueError):
            calibrate(data,1)

    def test_excessive_validation_error_is_rejected(self):
        data=synthetic_data()
        data['validation'][1]['mm'][0]+=20
        with self.assertRaises(ValueError):
            calibrate(data,1)

    def test_degenerate_points_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_homography([[0,0],[1,1],[2,2],[3,3]],[[0,0],[1,1],[2,2],[3,3]])

    def test_nonfinite_or_zero_projection_is_rejected(self):
        with self.assertRaises(ValueError):
            project([[1,0,0],[0,1,0],[0,0,0]],[[1,1]])
        with self.assertRaises(ValueError):
            calibrate(synthetic_data(),float('nan'))


if __name__=='__main__':
    unittest.main()
