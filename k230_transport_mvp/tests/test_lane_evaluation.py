import unittest
from tools.evaluate_lane_sequences import evaluate


class LaneEvaluationTests(unittest.TestCase):
    def test_offsets_and_rejections_keep_exact_denominators(self):
        report=evaluate()
        self.assertEqual(report['offset']['expected'],9)
        self.assertEqual(report['offset']['passed'],9)
        self.assertLessEqual(max(report['offset']['errors_px']),2)
        self.assertEqual(report['rejection'],{'expected':9,'passed':9})
        self.assertEqual(report['boundary'],{'expected':18,'passed':18})
        self.assertTrue(report['source_files_unchanged'])
        self.assertEqual(report['status'],'PASS')

    def test_thin_roi_case_is_retained_and_not_camera_evidence(self):
        report=evaluate();row=next(r for r in report['cases'] if r['input']['name']=='one-unique-row')
        self.assertEqual(row['unique_sample_rows'],1)
        self.assertTrue(row['behavior_passed'])
        self.assertTrue(report['is_simulated']);self.assertFalse(report['live_CSI2_lane_acceptance'])
        self.assertFalse(report['physical_camera_accessed']);self.assertFalse(report['motors_enabled'])
