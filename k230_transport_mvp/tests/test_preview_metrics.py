import unittest
from k230.preview_metrics import PreviewMetrics


class PreviewMetricsTests(unittest.TestCase):
    def test_interval_delta_and_capture_pause_are_not_cumulative_average(self):
        m=PreviewMetrics()
        m.update(('TOP',5),0,{'frames':0,'display_frames':0,'capture_jobs':0})
        self.assertIsNone(m.update(('TOP',5),999,{'frames':30,'display_frames':15,'capture_jobs':0}))
        first=m.update(('TOP',5),1000,{'frames':30,'display_frames':15,'capture_jobs':0})
        self.assertEqual(first['processing_fps'],30)
        paused=m.update(('TOP',5),3000,{'frames':50,'display_frames':25,'capture_jobs':1})
        self.assertEqual(paused['processing_fps'],10)
        self.assertEqual(paused['display_submit_fps'],5)
        self.assertTrue(paused['contains_synchronous_capture'])

    def test_first_and_tail_frames_are_accounted_without_inventing_full_interval(self):
        m=PreviewMetrics()
        m.reset(('TOP',5),0,{'frames':0})
        m.update(('TOP',5),1000,{'frames':30})
        tail=m.finish(1500,{'frames':45})
        self.assertFalse(tail['complete_interval'])
        self.assertEqual(sum(w['delta']['frames'] for w in m.windows),45)
        self.assertEqual(m.discarded_partial_frames,0)

    def test_context_change_discards_partial_and_bounds_retention(self):
        m=PreviewMetrics(max_windows=2)
        m.update(('TOP',5),0,{'frames':0})
        m.update(('TOP',1),500,{'frames':15})
        self.assertEqual(m.discarded_partial_windows,1)
        self.assertIsNone(m.update(('TOP',1),1499,{'frames':40}))
        for now,frames in ((1500,45),(2500,75),(3500,105)):
            row=m.update(('TOP',1),now,{'frames':frames})
            self.assertEqual(row['processing_fps'],30)
        self.assertEqual(len(m.windows),2)
        self.assertTrue(all(w['context']==('TOP',1) for w in m.windows))
