import unittest

import numpy as np

from neural_decode import decode_logits


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.centers = [[x, y] for y in range(-1175, 1200, 50) for x in range(-1175, 1200, 50)]

    def test_temperature_scoring_and_hold_match_numpy(self):
        values = np.linspace(-8, 8, 2304)
        decoded = decode_logits(values.tolist(), self.centers, 2., None)
        probability = np.exp((values - values.max()) / 2.)
        self.assertAlmostEqual(decoded['visual_peak_mass'], float(probability[-1] / probability.sum()))
        self.assertEqual(decoded['top_k_camera_xy'][0], dict(camera_x_mm=1175, camera_y_mm=1175))
        self.assertIsNone(decoded['camera_x_mm'])
        self.assertEqual(decoded['status'], 'unlocalized')

    def test_flat_ties_are_stable_and_nonfinite_is_rejected(self):
        answer = decode_logits([0.] * 2304, self.centers, 1.)
        self.assertEqual(answer['top_k_camera_xy'][0], dict(camera_x_mm=-1175, camera_y_mm=-1175))
        bad = [0.] * 2304
        bad[9] = float('nan')
        with self.assertRaises(ValueError):
            decode_logits(bad, self.centers, 1.)
