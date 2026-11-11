import math
import unittest
import cv2
import numpy as np
from unittest.mock import patch
from vision.ellipse_native import extract_mask_ellipses, contour_fit, ellipse_outline


def analytic_mask(width=220, height=160, ring=False):
    y, x = np.mgrid[:height, :width]
    radius = ((x-90)/35)**2+((y-70)/24)**2
    return ((radius <= 1) & ((radius > .65) if ring else True)).astype("uint8")*255


class NativeEllipseTests(unittest.TestCase):
    def test_vector_backend_preserves_full_boundary_acceptance_and_metrics(self):
        from vision import ellipse_native as native
        rng=np.random.default_rng(42)
        for count in (48,160,512,1024):
            for angle in (0,.35,1.1):
                for noise in (0,.3,1.5,4):
                    theta=np.linspace(0,2*math.pi,count,endpoint=False)
                    u=35*np.cos(theta);v=24*np.sin(theta)
                    pts=np.stack((90+u*math.cos(angle)-v*math.sin(angle),70+u*math.sin(angle)+v*math.cos(angle)),axis=1)
                    pts+=rng.normal(0,noise,pts.shape)
                    for tail in (False,True):
                        points=pts.tolist()
                        if tail:points[count//2]=[160,70]
                        with patch.object(native,'_BOUNDARY_NP',None):scalar=contour_fit(points)
                        with patch.object(native,'_BOUNDARY_NP',np):vector=contour_fit(points)
                        self.assertEqual((scalar['accepted'],scalar['reason']),(vector['accepted'],vector['reason']))
                        for key in ('rms_px','max_error_px','inlier_fraction','contour_points','sectors'):
                            if key in scalar:self.assertAlmostEqual(scalar[key],vector[key],places=7)

    def test_vector_backend_rejects_degenerate_and_clipped_source_points(self):
        from vision import ellipse_native as native
        points=[[90+35*math.cos(i*2*math.pi/64),70+24*math.sin(i*2*math.pi/64)] for i in range(64)]
        with patch.object(native,'_BOUNDARY_NP',np):
            self.assertEqual(contour_fit(points,roi=[90,0,130,160])['reason'],'ROI_OR_IMAGE_CLIPPED')
            self.assertIsNone(native.vector_boundary_metrics([[0,0]]*16,0,0,10,10,1,0,np))
            self.assertIsNone(native.vector_boundary_metrics([[float('nan'),0]]*16,0,0,10,10,1,0,np))

    def test_native_fit_rotation_and_translation_match_analytic_centers(self):
        y,x=np.mgrid[:180,:220]
        for angle in (0,.35,1.1,2.4):
            u=(x-110)*math.cos(angle)+(y-90)*math.sin(angle)
            v=-(x-110)*math.sin(angle)+(y-90)*math.cos(angle)
            mask=((u/37)**2+(v/23)**2<=1).astype('uint8')*255
            result=extract_mask_ellipses(mask,[30,40,220,180],cv=cv2)
            self.assertEqual(len(result['accepted']),1,result)
            self.assertLess(math.dist(result['accepted'][0]['center'],[140,130]),.4)

    def test_close_kernel_is_bounded_and_fills_small_boundary_gaps(self):
        mask=analytic_mask();mask[68,90:95]=0
        without=extract_mask_ellipses(mask,[0,0,220,160],cv=cv2,close_kernel=0)
        with_close=extract_mask_ellipses(mask,[0,0,220,160],cv=cv2,close_kernel=3)
        self.assertEqual(with_close['close_kernel'],3)
        self.assertGreaterEqual(len(with_close['accepted']),len(without['accepted']))
        with self.assertRaises(ValueError):extract_mask_ellipses(mask,[0,0,220,160],cv=cv2,close_kernel=4)

    def test_empty_mask_returns_without_contour_scan(self):
        result=extract_mask_ellipses(np.zeros((160,220),dtype='uint8'),[0,0,220,160],cv=cv2,close_kernel=3)
        self.assertEqual(result['reason'],'NO_MATCH_PIXELS')
        self.assertEqual(result['components'],[])

    def test_complete_native_contour_has_source_center_and_full_boundary_metrics(self):
        result = extract_mask_ellipses(analytic_mask(), [10, 20, 220, 160], cv=cv2)
        self.assertEqual(len(result["accepted"]), 1, result)
        fit = result["accepted"][0]
        self.assertLess(math.dist(fit["center"], [100, 90]), .3)
        self.assertEqual(fit["contour_points"], fit["points"])
        self.assertEqual(fit["backend"], "NATIVE_CV2_FIT_ELLIPSE")
        self.assertLess(fit["max_error_px"], 2)
        self.assertEqual(len(ellipse_outline(fit)), 48)

    def test_ring_hole_and_clipped_contour_have_no_ellipse(self):
        result = extract_mask_ellipses(analytic_mask(ring=True), [0, 0, 220, 160], cv=cv2)
        self.assertEqual(result["accepted"], [], result)
        self.assertEqual(result["components"][0]["fit"]["reason"], "INTERIOR_FILL")
        clipped = analytic_mask()[:, 90:]
        result = extract_mask_ellipses(clipped, [90, 0, 130, 160], cv=cv2)
        self.assertEqual(result["accepted"], [], result)
        self.assertEqual(result["components"][0]["fit"]["reason"], "ROI_OR_IMAGE_CLIPPED")

    def test_ellipse_area_limit_uses_contour_area_not_enclosing_rectangle(self):
        result = extract_mask_ellipses(analytic_mask(), [0, 0, 220, 160], max_area=3000, cv=cv2)
        self.assertEqual(len(result["accepted"]), 1, result)
        self.assertEqual(result["area_semantics"], "NATIVE_CONTOUR_AREA")
        self.assertEqual(extract_mask_ellipses(analytic_mask(), [0, 0, 220, 160], max_area=2000, cv=cv2)["accepted"], [])

    def test_two_same_color_instances_and_loss_are_not_cached_or_nearest_selected(self):
        mask = analytic_mask()
        mask[90:130, 160:200] = 255
        result = extract_mask_ellipses(mask, [0, 0, 220, 160], cv=cv2)
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(len(result["components"]), 2)
        self.assertEqual(extract_mask_ellipses(mask*0, [0, 0, 220, 160], cv=cv2)["accepted"], [])

    def test_unsampled_tail_is_checked_against_full_boundary(self):
        points = [[90+35*math.cos(2*math.pi*i/1024), 70+24*math.sin(2*math.pi*i/1024)] for i in range(1024)]
        points[513] = [160, 70]
        fit = contour_fit(points)
        self.assertFalse(fit["accepted"])
        self.assertEqual(fit["reason"], "FULL_BOUNDARY_RESIDUAL")

    def test_merged_body_rejected_and_all_contours_remain_bounded(self):
        mask = analytic_mask()
        mask[70:142, 70:111] = 255
        result = extract_mask_ellipses(mask, [0, 0, 220, 160], cv=cv2)
        self.assertEqual(result["accepted"], [], result)
        points = [[i, i] for i in range(4097)]
        self.assertEqual(contour_fit(points)["reason"], "CONTOUR_POINT_LIMIT")


if __name__ == "__main__":
    unittest.main()
