import unittest
import cv2
import numpy as np
from unittest.mock import patch
from vision.ring_native import (extract_ring_edges,
    detect_paper_targets,associate_material_target,detect_paper_circle_proposals)


class NativeRingTests(unittest.TestCase):
    def test_gray_circle_proposals_keep_unverified_flags_and_need_two_stroke_levels(self):
        gray=np.full((220,220),200,dtype=np.uint8)
        for radius in (24,44):cv2.circle(gray,(100,100),radius,60,2)
        # Feed exact native proposals to test the independent stroke validation.
        circles=np.array([[[100,100,24],[100,100,44]]],np.float32)
        with patch.object(cv2,'HoughCircles',return_value=circles):
            result=detect_paper_circle_proposals(gray,(20,30,220,220),pixel_scale=1)
        self.assertEqual(len(result['groups']),1)
        target=result['groups'][0]
        self.assertEqual(target['center'],[120,130])
        self.assertTrue(target['proposal_only']);self.assertFalse(target['identity_verified'])
        self.assertTrue(all(not f['boundary_complete_verified'] for f in target['boundaries']))
        self.assertIsNone(associate_material_target([self.fit(120,130,10)],[target])['target'])
        for kind in ('single','solid','clipped'):
            scene=np.full((220,220),200,dtype=np.uint8)
            cv2.circle(scene,(100,100),24,60,-1 if kind=='solid' else 2)
            proposed=circles if kind!='clipped' else np.array([[[10,100,24],[10,100,44]]],np.float32)
            with self.subTest(kind=kind),patch.object(cv2,'HoughCircles',return_value=proposed):
                self.assertEqual(detect_paper_circle_proposals(scene,(0,0,220,220),pixel_scale=1)['groups'],[])

    def test_gray_circle_proposals_cap_output_and_preserve_empty_results(self):
        gray=np.full((220,220),200,dtype=np.uint8)
        with patch.object(cv2,'HoughCircles',return_value=None):
            self.assertEqual(detect_paper_circle_proposals(gray,(0,0,220,220))['groups'],[])
        with patch.object(cv2,'HoughCircles',return_value=np.zeros((1,129,3),np.float32)):
            result=detect_paper_circle_proposals(gray,(0,0,220,220))
        self.assertFalse(result['complete']);self.assertEqual(result['reason'],'PAPER_CIRCLE_PROPOSAL_LIMIT')

    def test_duplicate_circle_votes_on_one_dark_stroke_do_not_create_two_levels(self):
        scene=np.full((220,220),200,dtype=np.uint8)
        cv2.circle(scene,(100,100),50,60,2)
        votes=np.array([[[100,100,47],[100,100,53]]],np.float32)
        with patch.object(cv2,'HoughCircles',return_value=votes):
            result=detect_paper_circle_proposals(scene,(0,0,220,220),pixel_scale=1)
        self.assertEqual(len(result['boundaries']),2)
        self.assertEqual(result['groups'],[])
    def paper_scene(self,occupied=False):
        gray=np.full((240,440),200,dtype=np.uint8)
        for x in (110,330):
            for radius in (14,34,54):cv2.circle(gray,(x,120),radius,80,2)
        if occupied:cv2.circle(gray,(110,120),23,100,-1)
        return gray

    def test_paper_targets_detect_without_any_material_and_survive_occlusion(self):
        empty=detect_paper_targets(self.paper_scene(),(0,0,440,240))['groups']
        occupied=detect_paper_targets(self.paper_scene(True),(0,0,440,240))['groups']
        self.assertEqual((len(empty),len(occupied)),(2,2))
        for target,covered in zip(empty,occupied):
            self.assertLess(sum((a-b)**2 for a,b in zip(target['center'],covered['center'])),1)
        self.assertEqual(occupied[0]['support'],2)
        self.assertEqual(empty[0]['support'],3)

    def test_solid_material_disk_does_not_supply_paper_stroke_levels(self):
        gray=np.full((240,440),200,dtype=np.uint8)
        cv2.circle(gray,(110,120),23,100,-1)
        result=detect_paper_targets(gray,(0,0,440,240))
        self.assertEqual(result['groups'],[])
        self.assertEqual(result['boundaries'],[])

    def test_association_never_mutates_or_generates_paper_detections(self):
        import copy
        targets=detect_paper_targets(self.paper_scene(),(0,0,440,240))['groups']
        before=copy.deepcopy(targets)
        self.assertIsNone(associate_material_target([],targets)['target'])
        result=associate_material_target([self.fit(110,120,23)],targets)
        self.assertEqual(result['target']['center'],targets[0]['center'])
        self.assertIsNone(associate_material_target([self.fit(220,40,23)],targets)['target'])
        self.assertEqual(targets,before)

    def fit(self,x,y,r):
        return {'center':[x,y],'semiaxes':[r,r],'rms_px':.2}

    def test_concentric_targets_keep_source_centers_and_collapse_stroke_edges(self):
        gray=np.full((220,440),200,dtype=np.uint8)
        for center in ((110,110),(330,110)):
            for axes in ((16,13),(34,27),(58,46)):
                cv2.ellipse(gray,center,axes,20,0,360,100,2)
        result=extract_ring_edges(gray,(20,30,440,220))
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['groups']),2)
        for group,x in zip(result['groups'],(130,350)):
            self.assertAlmostEqual(group['center'][0],x,delta=2)
            self.assertAlmostEqual(group['center'][1],140,delta=2)
            self.assertEqual(group['support'],3)
            self.assertFalse(group['identity_verified'])

    def test_single_wheel_like_disk_stroke_and_separate_circles_are_not_targets(self):
        for kind in ('disk','stroke','separate'):
            gray=np.full((220,440),200,dtype=np.uint8)
            if kind=='disk':cv2.circle(gray,(200,110),50,70,-1)
            elif kind=='stroke':cv2.circle(gray,(200,110),50,70,2)
            else:
                for x in (90,220,350):cv2.circle(gray,(x,110),40,70,2)
            with self.subTest(kind=kind):
                self.assertEqual(extract_ring_edges(gray,(0,0,440,220))['groups'],[])

    def test_partial_clipped_target_never_supplies_complete_ring_levels(self):
        gray=np.full((220,220),200,dtype=np.uint8)
        for r in (20,40,60):cv2.circle(gray,(10,110),r,70,2)
        self.assertEqual(extract_ring_edges(gray,(0,0,220,220))['groups'],[])

    def test_coarse_preview_restores_native_coordinates_and_mask_size(self):
        gray=np.full((220,440),200,dtype=np.uint8)
        for r in (18,36,60):cv2.circle(gray,(200,110),r,80,2)
        result=extract_ring_edges(gray,(20,30,440,220),pixel_scale=2)
        self.assertEqual(result['pixel_scale'],2)
        self.assertEqual(result['edges'].shape,gray.shape)
        self.assertEqual(len(result['groups']),1)
        self.assertAlmostEqual(result['groups'][0]['center'][0],220,delta=2)
        self.assertAlmostEqual(result['groups'][0]['center'][1],140,delta=2)

    def test_edge_sensitivity_does_not_turn_one_outline_into_concentric_levels(self):
        gray=np.full((220,220),160,dtype=np.uint8)
        cv2.circle(gray,(110,110),50,145,1)
        self.assertEqual(extract_ring_edges(gray,(0,0,220,220),edge_thresholds=(5,15))['groups'],[])
        for value in ((0,15),(15,5),(True,15)):
            with self.assertRaises(ValueError):extract_ring_edges(gray,(0,0,220,220),edge_thresholds=value)


if __name__=='__main__':unittest.main()
