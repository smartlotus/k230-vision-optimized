import unittest
import cv2
import numpy as np
from unittest.mock import patch
from vision.ellipse_edges import radial_edge_fit,extract_dark_edge_ellipses
from vision.ellipse_native import extract_mask_ellipses


def fixture():
    y,x=np.mgrid[:160,:200]
    top=((x-100)/29)**2+((y-65)/27)**2<=1
    body=(abs(x-100)<=24)&(y>=65)&(y<=135)
    gray=np.full((160,200),150,dtype='uint8');gray[body]=15;gray[top]=50
    lab=(top|body).astype('uint8')*255
    return gray,lab,top,body


class EdgeEllipseTests(unittest.TestCase):
    def test_rounded_border_samples_are_rejected_before_array_reads(self):
        gray=np.zeros((100,100),np.uint8);gray[73,70:]=80;gray[89,66]=80
        lab=np.full((100,100),255,np.uint8)
        for proposal in ([50,72.75,20],[72.75,50,20],[72.5,50,20],[50,72.5,20]):
            with self.subTest(proposal=proposal):
                result=radial_edge_fit(gray,lab,proposal,[0,0,100,100],120,6000,cv2)
                self.assertEqual(result['reason'],'EDGE_SEARCH_CLIPPED')
    def test_spread_search_keeps_all_successful_points_and_rejects_broken_edges_earlier(self):
        from vision import ellipse_edges
        gray,lab,_,_=fixture()
        args=([100,65,28],[0,0,200,160],700,6000,cv2)
        with patch.object(ellipse_edges,'_EDGE_ORDER',tuple(range(48))):
            previous=radial_edge_fit(gray,lab,*args)
        current=radial_edge_fit(gray,lab,*args)
        self.assertEqual(current,previous)
        y,x=np.indices(gray.shape);angle=np.arctan2(y-65,x-100)%(2*np.pi)
        gray[(angle>2.8)&(angle<3.5)]=50
        class CountingGray:
            def __init__(self):self.shape=gray.shape;self.reads=0
            def __getitem__(self,key):self.reads+=1;return gray[key]
        first=CountingGray();second=CountingGray()
        with patch.object(ellipse_edges,'_EDGE_ORDER',tuple(range(48))):
            previous=radial_edge_fit(first,lab,*args)
        current=radial_edge_fit(second,lab,*args)
        self.assertEqual(current,previous);self.assertFalse(current['accepted'])
        self.assertLess(second.reads,first.reads)
    def test_distinct_top_edge_recovers_center_when_lab_includes_body(self):
        gray,lab,_,_=fixture()
        primary=extract_mask_ellipses(lab,[0,0,200,160],700,6000)
        self.assertEqual(primary['accepted'],[])
        result=radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)
        self.assertTrue(result['accepted'],result)
        self.assertLess(np.linalg.norm(np.array(result['center'])-[100,65]),1.5)
        self.assertEqual(result['edge_directions'],48)
        self.assertGreaterEqual(result['interior_fill'],.9)

    def test_native_proposals_recover_only_one_top_without_duplicate_identity(self):
        gray,lab,_,_=fixture()
        primary=extract_mask_ellipses(lab,[0,0,200,160],700,6000)
        result=extract_dark_edge_ellipses(gray,lab,[0,0,200,160],primary['components'],700,6000)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['accepted']),1,result)

    def test_uniform_merged_body_and_rectangle_do_not_become_round_from_prior(self):
        gray,lab,top,body=fixture();gray[top|body]=50
        self.assertFalse(radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])
        gray[:]=150;gray[38:93,73:128]=50;lab[:]=255
        self.assertFalse(radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])
        gray[:]=50
        self.assertFalse(radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])

    def test_missing_color_hole_partial_edge_and_clipping_are_rejected(self):
        gray,lab,_,_=fixture()
        self.assertFalse(radial_edge_fit(gray,lab*0,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])
        lab[55:75,90:110]=0
        self.assertFalse(radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])
        gray,lab,_,_=fixture();gray[:65,:100]=50
        self.assertFalse(radial_edge_fit(gray,lab,[100,65,28],[0,0,200,160],700,6000,cv2)['accepted'])
        self.assertFalse(radial_edge_fit(gray,lab,[20,65,28],[0,0,200,160],700,6000,cv2)['accepted'])


if __name__=='__main__':unittest.main()
