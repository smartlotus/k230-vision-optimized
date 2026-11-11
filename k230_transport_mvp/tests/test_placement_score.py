import unittest
from vision.placement_score import RING_POINTS,alignment_quality,score_ring_placement,score_placement_receipt


class PlacementScoreTests(unittest.TestCase):
    def receipt(self, **fields):
        value=dict(status='DONE',action='PLACE',placement_verified=True,
            score_observation_verified=True,is_simulated=True,slot=3,observed_slot=3,
            scoring_zone=1,placement_state='IN_RING',stable=True,tilted=False)
        value.update(fields);return value

    def test_all_eighteen_slot_and_zone_combinations_are_independent(self):
        for slot in (1,2,3):
            for zone,points in RING_POINTS.items():
                r=score_ring_placement(slot,slot,scoring_zone=zone,
                    placement_state='IN_RING',stable=True,calibrated=True)
                self.assertEqual(r['official_points'],points)

    def test_slot_three_can_score_fifteen_and_zone_six_scores_one(self):
        self.assertEqual(score_placement_receipt(3,self.receipt(),calibrated=True)['official_points'],15)
        self.assertEqual(score_placement_receipt(3,self.receipt(scoring_zone=6),calibrated=True)['official_points'],1)

    def test_command_echo_is_not_postplacement_observed_slot_or_scoring_zone(self):
        for key in ('observed_slot','scoring_zone'):
            receipt=self.receipt();receipt.pop(key)
            self.assertIsNone(score_placement_receipt(3,receipt,calibrated=True)['official_points'])

    def test_wrong_observed_slot_is_zero_even_with_matching_command_echo(self):
        r=score_placement_receipt(3,self.receipt(observed_slot=2),calibrated=True)
        self.assertEqual((r['official_points'],r['reason']),(0,'WRONG_SLOT'))

    def test_ack_pick_or_missing_score_proof_stays_unknown(self):
        for fields in ({'status':'ACK'},{'action':'PICK'},{'placement_verified':False},
                       {'score_observation_verified':False},{'is_simulated':None}):
            self.assertIsNone(score_placement_receipt(3,self.receipt(**fields),calibrated=True)['official_points'])

    def test_uncalibrated_or_unstable_stays_unknown(self):
        self.assertIsNone(score_placement_receipt(3,self.receipt())['official_points'])
        self.assertIsNone(score_placement_receipt(3,self.receipt(stable=False),calibrated=True)['official_points'])

    def test_confirmed_outside_fell_and_tilted_are_zero(self):
        for fields in ({'placement_state':'OUTSIDE'},{'placement_state':'FELL'},{'tilted':True}):
            self.assertEqual(score_placement_receipt(3,self.receipt(**fields))['official_points'],0)

    def test_simulation_provenance_is_retained(self):
        self.assertTrue(score_placement_receipt(3,self.receipt(),calibrated=True)['is_simulated'])

    def test_pixel_quality_is_separate_and_nonfinite_or_boolean_values_rejected(self):
        self.assertEqual(alignment_quality(2,10,stable=True)['quality'],80)
        self.assertEqual(alignment_quality(None,10,stable=True)['status'],'UNKNOWN')
        self.assertEqual(alignment_quality(2,10,stable=False)['status'],'REJECTED')
        for error,tolerance in [(True,10),(1,False),(float('nan'),10),(1,float('inf'))]:
            with self.assertRaises(ValueError):alignment_quality(error,tolerance,stable=True)


if __name__=='__main__':unittest.main()
