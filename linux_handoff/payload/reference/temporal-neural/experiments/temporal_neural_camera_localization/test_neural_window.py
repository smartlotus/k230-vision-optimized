import unittest

import numpy as np

from neural_window import FrameWindow


class NeuralWindowTests(unittest.TestCase):
    def test_oldest_padding_ring_order_and_gap_reset(self):
        window = FrameWindow(np, 4, 388, 700, lambda current, before: current - before)
        for i, expected in enumerate(((1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 3),
                                      (1, 2, 3, 4), (2, 3, 4, 5))):
            rgb = np.full((1, 3, 96, 160), i + 1, dtype=np.uint8)
            packed = window.append(rgb, i * 400)
            self.assertEqual(tuple(packed[0, ::3, 0, 0]), expected)
            self.assertFalse(window.due(i * 400 + 387))
        packed = window.append(np.full((1, 3, 96, 160), 9, dtype=np.uint8), 2500)
        self.assertEqual(tuple(packed[0, ::3, 0, 0]), (9, 9, 9, 9))
        self.assertEqual((window.count, window.resets), (1, 1))

    def test_wrapping_clock_uses_injected_difference(self):
        wrap = 1 << 16
        tick_difference = lambda new, old: (new - old + wrap // 2) % wrap - wrap // 2
        window = FrameWindow(np, 8, 388, 700, tick_difference)
        rgb = np.ones((1, 3, 96, 160), dtype=np.uint8)
        window.append(rgb, wrap - 50)
        self.assertFalse(window.due(330))
        window.append(rgb * 2, 350)
        self.assertEqual(tuple(window.tensor[0, ::3, 0, 0]), (1, 1, 1, 1, 1, 1, 1, 2))

    def test_sixteen_frame_window_preserves_rgb_order_and_timestamps(self):
        window = FrameWindow(np, 16, 230000, 730405, lambda current, before: current - before)
        for i in range(20):
            window.append(np.full((1, 3, 96, 160), i, dtype=np.uint8), i * 250000)
        self.assertEqual(window.tensor.shape, (1, 48, 96, 160))
        self.assertEqual(window.tensor[0, ::3, 0, 0].tolist(), list(range(4, 20)))
        self.assertEqual(window.tensor_timestamps_us, [i * 250000 for i in range(4, 20)])
