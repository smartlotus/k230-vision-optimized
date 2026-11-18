import unittest
import numpy as np
from temporal_filter import TemporalCameraFilter


class FilterTests(unittest.TestCase):
    def point(self, row, col, size=24):
        p = np.zeros((size, size))
        p[row, col] = 1
        return p

    def test_known_bound_removes_far_peak_and_empty_topk(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True)
        f.initialize(self.point(5, 5), 0.)
        vis = .6 * self.point(5, 6) + .4 * self.point(18, 18)
        result = f.update(vis, .1, 300., 0.)
        self.assertEqual(result['state'], 'TRACKING')
        self.assertEqual(result['camera_x_mm'], -550.)
        self.assertEqual(len(result['top_k']), 1)

    def test_unknown_bounds_leave_visual_candidates_available(self):
        f = TemporalCameraFilter(np.ones((24, 24)))
        f.initialize(self.point(5, 5), 0.)
        result = f.update(self.point(18, 18), .1)
        self.assertEqual(result['camera_x_mm'], 650.)
        self.assertEqual(result['state'], 'DEGRADED')

    def test_initial_region_without_timestamp_is_preserved_on_first_update(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True)
        f.initialize(self.point(5, 5))
        result = f.update(.6 * self.point(5, 5) + .4 * self.point(18, 18), 10., 600., 0.)
        self.assertEqual(result['camera_x_mm'], -650.)
        self.assertEqual(result['dt_s'], 0.)

    def test_invalid_timestamp_is_atomic(self):
        f = TemporalCameraFilter(np.ones((24, 24)))
        f.initialize(self.point(5, 5), 1.)
        for value in [float('nan'), float('inf'), .5]:
            with self.assertRaises(ValueError):
                f.update(self.point(18, 18), value)
            self.assertEqual(f.last_ts, 1.)
            np.testing.assert_array_equal(f.belief, self.point(5, 5))

    def test_loss_then_recovery_changes_coordinate(self):
        f = TemporalCameraFilter(np.ones((24, 24)), bad_updates_to_lost=3, hard_gate_enabled=True)
        f.initialize(self.point(5, 5), 0.)
        for i in range(3):
            result = f.update(None, (i + 1) * .1, 0., 0.)
        self.assertEqual(result['state'], 'LOST')
        for i in range(3):
            result = f.update(self.point(18, 18), .4 + i * .1, 0., 0.)
        self.assertTrue(result['relocalized'])
        self.assertEqual(result['camera_x_mm'], 650.)

    def test_global_recovery_mass_cannot_escape_normal_reachable_update(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True)
        f.initialize(self.point(5, 5), 0.)
        vis = .1 * self.point(5, 5) + .9 * self.point(18, 18)
        for i in range(6):
            result = f.update(vis, (i + 1) * .1, 0., 0.)
            self.assertEqual(result['camera_x_mm'], -650.)
            self.assertEqual(f.belief[18, 18], 0.)
            self.assertFalse(result['relocalized'])

    def test_short_visual_loss_keeps_motion_prior_on_visibility_return(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True, bad_updates_to_lost=3)
        f.initialize(self.point(5, 5), 0.)
        for i in range(3):
            f.update(None, (i + 1) * .1, 0., 0.)
        result = f.update(.4 * self.point(5, 5) + .6 * self.point(18, 18), .4, 0., 0.)
        self.assertEqual(result['camera_x_mm'], -650.)
        self.assertFalse(result['relocalized'])

    def test_incoherent_far_candidates_require_new_confirmation_streak(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True, bad_updates_to_lost=3)
        f.initialize(self.point(5, 5), 0.)
        f.update(self.point(18, 18), .1, 0., 0.)
        result = f.update(self.point(1, 18), .2, 0., 0.)
        self.assertEqual(result['recovery_updates'], 1)
        self.assertFalse(result['relocalized'])
        for i in range(2):
            result = f.update(self.point(1, 18), .3 + .1 * i, 0., 0.)
        self.assertTrue(result['relocalized'])
        self.assertEqual(result['state'], 'DEGRADED')

    def test_edge_diffusion_conserves_mass_and_excludes_mask(self):
        mask = np.ones((24, 24), bool)
        mask[0, 1] = False
        f = TemporalCameraFilter(mask)
        f.initialize(self.point(0, 0), 0.)
        predicted = f.predict(.1, 500., 0.)
        self.assertAlmostEqual(predicted.sum(), 1.)
        self.assertEqual(predicted[0, 1], 0.)
        self.assertAlmostEqual(predicted[0, 0], 1/3)
        self.assertTrue(np.isfinite(f.predict(100., 500., 0.)).all())

    def test_reset_without_timestamp_and_long_gap(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True)
        f.reset_global()
        f.update(self.point(5, 5), 2., 300., 0.)
        result = f.update(self.point(18, 18), 20., 300., 0.)
        self.assertFalse(result['relocalized'])
        self.assertEqual(result['state'], 'DEGRADED')
        self.assertEqual(result['dt_s'], 18.)

    def test_long_gap_cannot_override_established_zero_motion(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True)
        f.initialize(self.point(5, 5), 0.)
        result = f.update(self.point(18, 18), 10., 0., 0.)
        self.assertFalse(result['relocalized'])
        self.assertEqual(result['camera_x_mm'], -650.)
        self.assertEqual(result['recovery_updates'], 1)

    def test_promoting_confirmed_mode_excludes_distant_secondary_mode(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True, bad_updates_to_lost=3)
        f.initialize(self.point(5, 5), 0.)
        visual = .9 * self.point(18, 18) + .1 * self.point(1, 18)
        for i in range(3):
            result = f.update(visual, (i + 1) * .1, 0., 0.)
        self.assertTrue(result['relocalized'])
        self.assertEqual(f.belief[1, 18], 0.)
        result = f.update(self.point(1, 18), .4, 0., 0.)
        self.assertFalse(result['relocalized'])
        self.assertEqual(result['camera_y_mm'], 650.)

    def test_zero_recovery_mix_can_restart_at_disjoint_candidate(self):
        f = TemporalCameraFilter(np.ones((24, 24)), hard_gate_enabled=True, global_recovery_mix=0.)
        f.initialize(self.point(5, 5), 0.)
        f.update(self.point(18, 18), .1, 0., 0.)
        result = f.update(self.point(1, 18), .2, 0., 0.)
        self.assertEqual(result['recovery_updates'], 1)
        self.assertFalse(result['relocalized'])

    def test_bad_likelihood_is_atomic(self):
        f = TemporalCameraFilter(np.ones((24, 24)))
        f.initialize(self.point(5, 5), 0.)
        for value in [-1., float('nan')]:
            vis = self.point(18, 18)
            vis[0, 0] = value
            with self.assertRaises(ValueError):
                f.update(vis, .1)
            self.assertEqual(f.last_ts, 0.)


if __name__ == '__main__':
    unittest.main()
