import unittest
from vision.placement_demo import VisualPlacementDemo,image_footprint_contained


def material(x=100,y=100,r=20):return {'center':[x,y],'semiaxes':[r,r]}
def paper(x=100,y=100,r=50):
    return {'center':[x,y],'boundaries':[{'center':[x,y],'semiaxes':[r,r]}],'identity_verified':False}


class VisualPlacementDemoTests(unittest.TestCase):
    def tick(self,demo,n,materials=None,papers=None):
        packet={'camera_id':2,'frame_id':n,'timestamp_ms':n*100,'is_simulated':True}
        return demo.update(packet,materials if materials is not None else [material()],papers or [],n*100)

    def holding(self,demo):
        for n in (1,2,3):result=self.tick(demo,n)
        self.assertEqual(result['state'],'SIM_HOLDING')
        return result

    def test_visible_material_confirms_simulated_grab_without_paper_or_motion(self):
        demo=VisualPlacementDemo();result=self.holding(demo)
        self.assertEqual(result['simulated_picks'],1)
        self.assertEqual(result['simulated_places'],0)
        self.assertEqual(result['events'][0]['event'],'SIM_PICK')
        self.assertFalse(result['motion_authorized']);self.assertFalse(result['physical_grab_verified'])

    def test_full_image_containment_confirms_one_simulated_place_and_latches(self):
        demo=VisualPlacementDemo();self.holding(demo)
        for n in (4,5,6):result=self.tick(demo,n,papers=[paper()])
        self.assertEqual(result['state'],'SIM_PLACED')
        self.assertEqual(result['events'][0]['event'],'SIM_PLACE')
        for n in range(7,20):
            result=self.tick(demo,n,papers=[paper()])
            self.assertEqual((result['simulated_picks'],result['simulated_places']),(1,1))
            self.assertEqual(result['events'],[])
        self.assertIsNone(result['official_points']);self.assertFalse(result['physical_placement_verified'])

    def test_center_inside_but_footprint_overhang_is_not_a_placement(self):
        self.assertFalse(image_footprint_contained(material(x=140,r=20),paper()))
        demo=VisualPlacementDemo();self.holding(demo)
        for n in range(4,10):r=self.tick(demo,n,[material(x=140)],papers=[paper()])
        self.assertEqual(r['simulated_places'],0)

    def test_missing_material_or_paper_does_not_complete_place(self):
        demo=VisualPlacementDemo();self.holding(demo)
        self.tick(demo,4,papers=[paper()]);self.tick(demo,5,papers=[paper()])
        self.tick(demo,6,materials=[],papers=[paper()])
        r=self.tick(demo,7,papers=[paper()])
        self.assertEqual(r['simulated_places'],0)
        self.assertEqual(self.tick(demo,8)['simulated_places'],0)

    def test_ambiguous_targets_do_not_choose_a_target(self):
        demo=VisualPlacementDemo();self.holding(demo)
        for n in range(4,10):r=self.tick(demo,n,papers=[paper(),paper(x=102)])
        self.assertEqual(r['simulated_places'],0)

    def test_unverified_paper_proposal_cannot_trigger_simulated_placement(self):
        for kind in ('target','boundary','incomplete_boundary'):
            target=paper()
            if kind=='target':target['proposal_only']=True
            elif kind=='boundary':target['boundaries'][0]['proposal_only']=True
            else:target['boundaries'][0]['boundary_complete_verified']=False
            self.assertFalse(image_footprint_contained(material(),target))
            demo=VisualPlacementDemo();self.holding(demo)
            for n in range(4,12):result=self.tick(demo,n,papers=[target])
            self.assertEqual(result['simulated_places'],0)

    def test_replayed_frame_cannot_confirm_grab_and_reset_starts_new_demo(self):
        demo=VisualPlacementDemo()
        for _ in range(10):r=self.tick(demo,1)
        self.assertEqual(r['simulated_picks'],0)
        self.tick(demo,2);self.tick(demo,3)
        demo.reset();self.assertEqual(demo.snapshot()['state'],'WAIT_MATERIAL')
        self.assertEqual(demo.snapshot()['simulated_picks'],0)

    def test_simulated_grab_precedes_place_even_if_material_starts_inside(self):
        demo=VisualPlacementDemo()
        for n in (1,2,3):r=self.tick(demo,n,papers=[paper()])
        self.assertEqual((r['simulated_picks'],r['simulated_places']),(1,0))
        for n in (4,5,6):r=self.tick(demo,n,papers=[paper()])
        self.assertEqual((r['simulated_picks'],r['simulated_places']),(1,1))


if __name__=='__main__':unittest.main()
