import copy
import unittest
from vision.block_detector import BlockDetector
from k230.block_profile import PARAMETERS


class Blob:
    def __init__(self,code,box,pixels):self.value=(code,box,pixels)
    def code(self):return self.value[0]
    def rect(self):return self.value[1]
    def pixels(self):return self.value[2]


class Image:
    def __init__(self,blobs=()):self.blobs=blobs;self.calls=[]
    def width(self):return 320
    def height(self):return 240
    def find_blobs(self,thresholds,**kwargs):self.calls.append((thresholds,kwargs));return self.blobs


class BlockTests(unittest.TestCase):
    def setUp(self):
        parameters=copy.deepcopy(PARAMETERS)
        parameters['roi']=(0,0,320,240)
        self.detector=BlockDetector(parameters)

    def test_one_native_pass_all_colors_centers_are_bbox_midpoints(self):
        image=Image([Blob(1,(10,20,30,50),1100),Blob(2,(60,20,30,50),1100),
                     Blob(4,(110,20,30,50),1100),Blob(32,(160,20,30,50),1100)])
        result=self.detector.process(image,1,100)
        self.assertEqual([r['target_id'] for r in result],[1,2,3,6])
        self.assertEqual(len(image.calls),1)
        self.assertFalse(image.calls[0][1]['merge'])
        self.assertEqual(result[0]['center'],(25,45))
        self.assertTrue(all(r['units']=='PX' and r['x_mm'] is None for r in result))

    def test_noise_density_multicode_overlap_and_oversized_are_rejected(self):
        blobs=[Blob(3,(10,10,30,50),1100),Blob(1,(60,10,30,50),100),
               Blob(1,(0,0,320,240),70000),Blob(8,(110,10,30,50),1100),
               Blob(4,(112,10,30,50),1100)]
        self.assertEqual(self.detector.detect(Image(blobs)),[])

    def test_clipped_can_be_seen_but_never_stable_eligible(self):
        image=Image([Blob(4,(290,30,30,60),1300)])
        for i in range(1,5):result=self.detector.process(image,i,i*30)
        self.assertTrue(result[0]['clipped'])
        self.assertFalse(result[0]['eligible'])
        self.assertFalse(result[0]['stable'])

    def test_lost_targets_disappear_and_reappearance_needs_new_confirmation(self):
        image=Image([Blob(1,(30,30,30,50),1100)])
        for i in range(1,4):result=self.detector.process(image,i,i*30)
        self.assertTrue(result[0]['stable'])
        previous=result[0]['track_id']
        self.assertEqual(self.detector.process(Image(),4,120),[])
        result=self.detector.process(image,5,150)
        self.assertNotEqual(result[0]['track_id'],previous)
        self.assertFalse(result[0]['stable'])

    def test_two_same_color_instances_keep_separate_tracks(self):
        image=Image([Blob(1,(20,30,30,50),1100),Blob(1,(120,30,30,50),1200)])
        for i in range(1,4):result=self.detector.process(image,i,i*30)
        self.assertEqual(len({r['track_id'] for r in result}),2)
        self.assertTrue(all(r['stable'] for r in result))

    def test_replayed_out_of_order_or_long_gap_cannot_complete_stability(self):
        image=Image([Blob(1,(30,30,30,50),1100)])
        for ids,times in (([1,1,1],[10,20,30]),([3,2,1],[10,20,30]),([1,2,3],[10,500,1000])):
            self.detector.reset()
            for fid,ts in zip(ids,times):result=self.detector.process(image,fid,ts)
            self.assertFalse(result[0]['stable'])

    def test_motion_window_and_wrong_resolution(self):
        for i in range(1,5):
            result=self.detector.process(Image([Blob(1,(i*10,30,30,50),1100)]),i,i*30)
        self.assertFalse(result[0]['stable'])
        image=Image();image.width=lambda:640
        with self.assertRaisesRegex(ValueError,'resolution'):self.detector.detect(image)

    def test_unsampled_colors_have_no_enabled_threshold(self):
        parameters=copy.deepcopy(PARAMETERS)
        parameters['thresholds'][4]=parameters['thresholds'][5]=None
        detector=BlockDetector(parameters)
        self.assertNotIn(4,detector.color_ids)
        self.assertNotIn(5,detector.color_ids)

    def test_black_shadow_shape_filter_rejects_wide_low_density_patches(self):
        result=self.detector.detect(Image([Blob(16,(20,120,100,45),2900),
            Blob(16,(130,50,45,80),2800),Blob(16,(220,80,20,25),270),
            Blob(16,(16,141,30,22),500)]))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['target_id'],5)
        self.assertEqual(result[0]['bbox'],(130,50,45,80))

    def test_partial_overlap_retains_diagnostics_but_withholds_stability(self):
        image=Image([Blob(1,(20,30,40,70),2100),Blob(32,(50,30,40,70),2100)])
        for i in range(1,4):result=self.detector.process(image,i,i*40)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['overlapping'] for r in result))
        self.assertTrue(all(not r['stable'] and not r['eligible'] for r in result))

    def test_measured_black_component_below_old_density_cutoff_is_retained(self):
        result=self.detector.detect(Image([Blob(16,(140,60,45,73),2221)]))
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0]['density'],0.6761035,places=6)
        self.assertTrue(result[0]['eligible'])

    def test_measured_flat_cyan_reflection_is_not_an_upright_block(self):
        result=self.detector.detect(Image([Blob(32,(234,37,38,24),469),
            Blob(32,(95,50,44,84),2788)]))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['bbox'],(95,50,44,84))

    def test_profile_observation_window_boundary_is_not_a_complete_target(self):
        detector=BlockDetector(copy.deepcopy(PARAMETERS))
        image=Image([Blob(1,(20,35,30,60),1300)])
        for i in range(1,4):result=detector.process(image,i,i*40)
        self.assertFalse(result[0]['eligible'])
        self.assertFalse(result[0]['stable'])
        self.assertTrue(result[0]['clipped'])

    def test_measured_miss_keeps_dark_body_and_rejects_nearby_background(self):
        result=self.detector.detect(Image([Blob(16,(139,64,46,72),2103),
            Blob(16,(32,106,25,19),336)]))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['bbox'],(139,64,46,72))

    def test_small_measured_background_does_not_disable_complete_red(self):
        result=self.detector.detect(Image([Blob(1,(46,69,47,77),2809),
            Blob(16,(33,94,20,34),423),Blob(16,(138,71,45,72),2247)]))
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['eligible'] and not r['overlapping'] for r in result))
        self.assertEqual({r['target_id'] for r in result},{1,5})


if __name__=='__main__':unittest.main()
