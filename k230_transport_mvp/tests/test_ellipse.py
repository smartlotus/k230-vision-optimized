import math
import random
import unittest
from vision.ellipse import fit_ellipse
from vision.ellipse_segmentation import segment_lab_grid


def outline(cx, cy, major, minor, angle, count=96, arc=2*math.pi, noise=0):
    rng = random.Random(4107)
    points = []
    for i in range(count):
        t = arc*i/count
        x, y = major*math.cos(t), minor*math.sin(t)
        points.append([cx+x*math.cos(angle)-y*math.sin(angle)+rng.uniform(-noise, noise),
                       cy+x*math.sin(angle)+y*math.cos(angle)+rng.uniform(-noise, noise)])
    return points


def grid(shapes, width=96, height=80, stride=1):
    return {'source': 'REAL_CAMERA_NATIVE_LAB', 'is_simulated': False, 'camera_id': 1,
            'width': width, 'height': height, 'x0': 0, 'y0': 0, 'stride': stride,
            'lab_grid': [[[50, -30, 30] if any(shape(x, y) for shape in shapes) else [80, 30, -30]
                          for x in range(0, width, stride)] for y in range(0, height, stride)]}


THRESHOLD = [40, 60, -40, -20, 20, 40]


class EllipseTests(unittest.TestCase):
    def test_independent_analytic_rotated_ellipses_and_translated_coordinates(self):
        for angle in (0, .3, .9, 1.7, 2.8):
            for center in ((50, 40), (10000, -20000)):
                r = fit_ellipse(outline(*center, 30, 17, angle))
                self.assertTrue(r['accepted'], r)
                self.assertAlmostEqual(r['center'][0], center[0], places=6)
                self.assertAlmostEqual(r['center'][1], center[1], places=6)
                self.assertAlmostEqual(r['semiaxes'][0], 30, places=6)
                self.assertAlmostEqual(r['semiaxes'][1], 17, places=6)
                self.assertLess(abs(math.sin(r['angle_rad']-angle)), 1e-6)

    def test_noisy_contours_and_order_do_not_move_center_materially(self):
        points = outline(310, 180, 40, 28, .7, noise=.7)
        for p in (points, list(reversed(points)), points[37:]+points[:37]):
            r = fit_ellipse(p)
            self.assertTrue(r['accepted'], r)
            self.assertLess(math.dist(r['center'], [310, 180]), .4)

    def test_partial_arc_line_duplicate_nonfinite_and_unbounded_input_reject(self):
        cases = [outline(40, 40, 20, 15, 0, arc=math.pi), [[x, x] for x in range(30)],
                 [[5, 5]]*30, [[float('nan'), 3]]*30, outline(40, 40, 20, 15, 0, count=300)]
        for points in cases:
            r = fit_ellipse(points)
            self.assertFalse(r['accepted'], r)
            self.assertIsNone(r['center'])

    def test_rectangle_and_attached_tail_cannot_be_trimmed_into_an_ellipse(self):
        rectangle = [[x, y] for x in range(-40, 41, 4) for y in (-25, 25)]
        rectangle += [[x, y] for x in (-40, 40) for y in range(-21, 25, 4)]
        tail = outline(50, 40, 25, 20, 0, count=80)+[[80+i, 40] for i in range(20)]
        for points in (rectangle, tail):
            r = fit_ellipse(points)
            self.assertFalse(r['accepted'], r)
            self.assertIsNone(r['center'])

    def test_invalid_limits_raise_before_fitting(self):
        for kwargs in ({'max_points': 10000}, {'max_rms': float('inf')}, {'min_sectors': True}, {'max_ratio': .5}):
            with self.assertRaises(ValueError):
                fit_ellipse([], **kwargs)

    def test_two_separate_components_keep_two_centers_without_nearest_selection(self):
        s = grid([lambda x, y: ((x-25)/14)**2+((y-30)/10)**2 <= 1,
                  lambda x, y: ((x-70)/12)**2+((y-48)/16)**2 <= 1])
        r = segment_lab_grid(s, THRESHOLD, [0, 0, 96, 80])
        self.assertTrue(r['complete']);self.assertEqual(r['accepted'], 2, r)
        centers = sorted(c['fit']['center'] for c in r['components'])
        self.assertLess(math.dist(centers[0], [25, 30]), .5)
        self.assertLess(math.dist(centers[1], [70, 48]), .5)
        self.assertTrue(all(len(c['sampled_contour']) <= 128 for c in r['components']))

    def test_roi_clipping_is_rejected_even_when_remaining_curve_looks_elliptical(self):
        s = grid([lambda x, y: ((x-25)/14)**2+((y-30)/10)**2 <= 1])
        r = segment_lab_grid(s, THRESHOLD, [20, 0, 76, 80])
        self.assertEqual(r['accepted'], 0)
        self.assertEqual(r['components'][0]['fit']['reason'], 'ROI_OR_IMAGE_CLIPPED')

    def test_merged_body_fixture_is_rejected(self):
        s = grid([lambda x, y: ((x-40)/18)**2+((y-30)/14)**2 <= 1 or 29 <= x <= 51 and 30 <= y <= 66])
        r = segment_lab_grid(s, THRESHOLD, [0, 0, 96, 80])
        self.assertEqual(r['accepted'], 0, r)

    def test_texture_hole_is_not_fitted_as_an_inner_contour_but_ring_is_rejected(self):
        for hole_radius, expected in ((3, 1), (15, 0)):
            s = grid([lambda x, y: hole_radius**2 < (x-40)**2+(y-36)**2 <= 20**2])
            r = segment_lab_grid(s, THRESHOLD, [0, 0, 96, 80])
            self.assertEqual(r['accepted'], expected, r)
            self.assertGreater(r['components'][0]['interior_boundary_cells'], 0)
            if not expected:self.assertEqual(r['components'][0]['fit']['reason'], 'LOW_INTERIOR_FILL')

    def test_overflow_does_not_return_a_truncated_success_list(self):
        s = grid([lambda x, y: (x-20)**2+(y-30)**2 <= 100,
                  lambda x, y: (x-60)**2+(y-30)**2 <= 100])
        r = segment_lab_grid(s, THRESHOLD, [0, 0, 96, 80], max_components=1)
        self.assertFalse(r['complete']);self.assertEqual(r['components'], []);self.assertEqual(r['accepted'], 0)

    def test_sparse_grid_coordinates_and_invalid_source_binding(self):
        s = grid([lambda x, y: ((x-40)/20)**2+((y-36)/16)**2 <= 1], stride=4)
        r = segment_lab_grid(s, THRESHOLD, [0, 0, 96, 80], max_rms=3)
        self.assertEqual(r['accepted'], 1, r)
        self.assertLess(math.dist(r['components'][0]['fit']['center'], [40, 36]), 1)
        for key, value in (('camera_id', 2), ('is_simulated', True), ('source', 'JPEG_LAB')):
            bad = dict(s);bad[key] = value
            with self.assertRaises(ValueError):segment_lab_grid(bad, THRESHOLD, [0, 0, 96, 80])
        with self.assertRaises(ValueError):segment_lab_grid(s, [0, 100, -128, 127, 100, -100], [0, 0, 96, 80])


if __name__ == '__main__':
    unittest.main()
