import unittest
from tools.board_block_stability_check import BlockMetrics,Moments


def candidate(color=1,center=(10,20),track=1,stable=True,eligible=True):
    return {'target_id':color,'center':center,'track_id':track,'stable':stable,'eligible':eligible}


class MetricTests(unittest.TestCase):
    def test_online_moments_match_known_population_values(self):
        moments=Moments()
        self.assertIsNone(moments.report()['mean_px'])
        for point in ((1,10),(2,12),(3,14)):moments.add(point)
        r=moments.report()
        self.assertEqual(r['mean_px'],[2,12])
        self.assertAlmostEqual(r['population_std_px'][0],(2/3)**0.5)
        self.assertEqual(r['range_px'],[2,4])

    def test_unstable_samples_are_not_hidden_and_replay_does_not_increase_denominator(self):
        metrics=BlockMetrics()
        metrics.update([candidate(stable=False)],1)
        metrics.update([candidate(center=(13,24))],2)
        self.assertFalse(metrics.update([candidate(center=(100,100))],2))
        r=metrics.report();row=r['per_color']['1']
        self.assertEqual(r['frames'],2)
        self.assertEqual(r['replayed_frames_ignored'],1)
        self.assertEqual(row['eligible_centers']['samples'],2)
        self.assertEqual(row['stable_centers']['samples'],1)
        self.assertEqual(row['max_consecutive_step_px'],5)

    def test_ambiguous_or_ineligible_candidates_never_contribute_a_chosen_center(self):
        metrics=BlockMetrics()
        metrics.update([candidate(),candidate(center=(50,60),track=2)],1)
        metrics.update([candidate(eligible=False)],2)
        metrics.update([],3)
        row=metrics.report()['per_color']['1']
        self.assertEqual(row['multiple_candidate_frames'],1)
        self.assertEqual(row['multiple_eligible_frames'],1)
        self.assertEqual(row['eligible_centers']['samples'],0)
        self.assertEqual(row['longest_missing_run'],3)
        self.assertEqual(row['visible_frames'],2)

    def test_track_change_or_gap_does_not_count_as_consecutive_jitter(self):
        metrics=BlockMetrics()
        metrics.update([candidate()],1)
        metrics.update([],2)
        metrics.update([candidate(center=(100,100))],3)
        metrics.update([candidate(center=(200,200),track=2)],4)
        row=metrics.report()['per_color']['1']
        self.assertEqual(row['max_consecutive_step_px'],0)
        self.assertEqual(row['track_changes'],1)

    def test_all_six_metrics_use_the_same_frame_denominator(self):
        metrics=BlockMetrics()
        metrics.update([candidate(c) for c in range(1,7)],1)
        metrics.update([candidate(c,stable=c!=5) for c in range(1,7)],2)
        metrics.update([candidate(c) for c in range(1,6)],3)
        r=metrics.report()
        self.assertEqual(r['all_six_unique_eligible_frames'],2)
        self.assertEqual(r['all_six_unique_stable_frames'],1)
        self.assertEqual(r['per_color']['6']['longest_missing_run'],1)
        self.assertEqual(r['per_color']['5']['longest_unstable_run'],1)


if __name__=='__main__':unittest.main()
