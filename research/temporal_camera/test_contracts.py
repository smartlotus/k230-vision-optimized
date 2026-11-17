import unittest
import numpy as np
from filter import TemporalGridFilter
from dataset import schedule, audit, build_sequences
import copy


class FilterTests(unittest.TestCase):
    def test_unknown_motion_retains_global_support(self):
        f = TemporalGridFilter(np.ones((24,24),bool))
        v = np.full((24,24),.001);v[2,2]=1
        f.update(v,0)
        f.update(v,.1)
        self.assertTrue(np.all(f.belief>0))
        self.assertAlmostEqual(f.belief.sum(),1.)

    def test_lost_then_global_relocalizes(self):
        f=TemporalGridFilter(np.ones((24,24),bool),bad_limit=3)
        v=np.zeros((24,24));v[1,1]=1
        f.update(v,0)
        for i in range(3):
            r=f.update(v,(i+1)*.1,visual_quality=0)
        self.assertEqual(r['state'],'LOST');self.assertFalse(r['relocalized'])
        v[:]=0;v[22,22]=1
        r=f.update(v,.4)
        self.assertTrue(r['relocalized']);self.assertEqual(r['state'],'TRACKING')
        self.assertEqual(r['camera_x_mm'],1050)

    def test_timestamp_and_invalid_visual(self):
        f=TemporalGridFilter(np.ones((8,8),bool))
        f.update(np.ones((8,8)),1)
        for ts in (float('nan'),.9):
            with self.assertRaises(ValueError):f.update(np.ones((8,8)),ts)
        self.assertEqual(f.update(np.full((8,8),float('nan')),2)['state'],'DEGRADED')

    def test_hard_gate_requires_bounds_and_can_recover(self):
        f=TemporalGridFilter(np.ones((24,24),bool),hard_gate_enabled=True,bad_limit=2)
        v=np.zeros((24,24));v[1,1]=1
        f.update(v,0)
        with self.assertRaises(ValueError):f.update(v,.1)
        v[:]=0;v[22,22]=1
        f.update(v,.1,100,0)
        self.assertEqual(f.update(v,.2,100,0)['state'],'LOST')
        self.assertTrue(f.update(v,.3,100,0)['relocalized'])


class DatasetTests(unittest.TestCase):
    def test_split_leakage_rejected(self):
        rows=list(schedule('pilot'))
        audit(rows)
        bad=copy.deepcopy(rows)
        bad[1]['split']='test'
        with self.assertRaisesRegex(ValueError,'Split leakage'):audit(bad)

    def test_bad_target_and_timestamp_rejected(self):
        rows=list(schedule('smoke'))
        rows[0]['camera_xy_target']['camera_yaw_deg']=0
        with self.assertRaises(ValueError):audit(rows)
        rows=list(schedule('smoke'));rows[1]['timestamp_s']=float('nan')
        with self.assertRaises(ValueError):audit(rows)

    def test_window_uses_one_trajectory_and_last_target(self):
        rows=list(schedule('pilot'));windows=build_sequences(rows,8)
        self.assertEqual(len(windows),300)
        self.assertEqual(len({w['sequence_id'] for w in windows}),300)
        self.assertTrue(all(w['target_index']==7 for w in windows))


if __name__=='__main__':unittest.main()
