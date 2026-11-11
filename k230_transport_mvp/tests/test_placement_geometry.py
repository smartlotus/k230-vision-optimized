import math
import unittest
from vision.placement_geometry import classify_circular_projection
from vision.placement_score import score_placement_receipt


RADII=[41.5,44,47.5,52.5,57.5,62.5]


class ProjectionGeometryTests(unittest.TestCase):
    def measure(self,dx=0,dy=0,radius=40,**fields):
        return classify_circular_projection(dx,dy,radius,RADII,geometry_verified=True,**fields)

    def test_equal_centers_do_not_mean_highest_zone_for_larger_material(self):
        self.assertEqual(self.measure(radius=40)['scoring_zone'],1)
        self.assertEqual(self.measure(radius=43)['scoring_zone'],2)
        self.assertEqual(self.measure(radius=50)['scoring_zone'],4)

    def test_all_six_zones_and_outside_follow_entire_footprint(self):
        for offset,zone in [(0,1),(3,2),(6,3),(10,4),(15,5),(20,6)]:
            with self.subTest(offset=offset):
                r=self.measure(dx=offset)
                self.assertEqual(r['scoring_zone'],zone)
                self.assertIsNone(r['official_points'])
        r=self.measure(dx=24)
        self.assertEqual(r['placement_state'],'OUTSIDE')
        self.assertIsNone(r['scoring_zone'])

    def test_signed_xy_offsets_use_radial_distance_and_are_rotation_invariant(self):
        for dx,dy in [(3,4),(-3,4),(0,-5)]:
            r=self.measure(dx,dy)
            self.assertEqual(r['center_error'],5)
            self.assertEqual(r['farthest_extent'],45)
            self.assertEqual(r['scoring_zone'],3)

    def test_boundary_contact_or_uncertainty_cannot_choose_points(self):
        for radius in RADII:
            r=self.measure(radius=radius)
            self.assertEqual(r['reason'],'ZONE_BOUNDARY_UNCERTAIN')
            self.assertIsNone(r['scoring_zone'])
        self.assertEqual(self.measure(uncertainty=2)['status'],'UNKNOWN')
        self.assertEqual(self.measure(uncertainty=1)['scoring_zone'],1)

    def test_unverified_pixels_or_hand_drawn_boundaries_never_supply_zone(self):
        r=classify_circular_projection(0,0,40,RADII)
        self.assertEqual(r['reason'],'VERIFIED_PROJECTION_REQUIRED')
        self.assertEqual(r['placement_state'],'UNKNOWN')

    def test_malformed_nonfinite_boolean_and_unordered_inputs_are_rejected(self):
        for value in (True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.measure(dx=value)
        for radii in ([1,2],[1,2,3,4,5,5],[1,2,3,4,6,5],[1,2,3,4,5,math.inf]):
            with self.assertRaises(ValueError):
                classify_circular_projection(0,0,1,radii,geometry_verified=True)
        with self.assertRaises(ValueError):self.measure(radius=0)
        with self.assertRaises(ValueError):self.measure(uncertainty=-1)

    def test_candidate_zone_enters_score_only_with_independent_place_proof(self):
        geometry=self.measure(dx=3)
        receipt={'status':'DONE','action':'PLACE','placement_verified':True,
            'score_observation_verified':True,'is_simulated':True,'observed_slot':3,
            'scoring_zone':geometry['scoring_zone'],'placement_state':geometry['placement_state'],
            'stable':True,'tilted':False}
        self.assertEqual(score_placement_receipt(3,receipt,calibrated=True)['official_points'],10)
        receipt['score_observation_verified']=False
        self.assertIsNone(score_placement_receipt(3,receipt,calibrated=True)['official_points'])


if __name__=='__main__':unittest.main()
