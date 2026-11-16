"""Channel/layout regression for the firmware planar-display workaround."""
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent/'board'))
from topface_trial import interleave_rgb, preview_rgb565


class CameraLayoutTests(unittest.TestCase):
    def test_interleave_preserves_channels_coordinates_and_owns_memory(self):
        planar = np.arange(3*4*6, dtype=np.uint8).reshape(3, 4, 6)
        original = planar.copy()
        packed = interleave_rgb(planar, 4, 6)
        np.testing.assert_array_equal(packed, original.transpose(1, 2, 0))
        self.assertTrue(packed.flags.c_contiguous)
        packed[0, 0, 0] = 255
        np.testing.assert_array_equal(planar, original)

    def test_preview_wraps_packed_bytes_and_never_converts_planar_frame(self):
        planar = np.zeros((3, 4, 6), dtype=np.uint8)
        planar[0], planar[1], planar[2] = 11, 79, 203
        frame = SimpleNamespace(height=lambda: 4, width=lambda: 6,
                                to_numpy_ref=lambda: planar,
                                to_rgb565=lambda: self.fail('Planar conversion is unsafe'))
        observed = {}
        result = object()

        def packed_image(width, height, format, *, alloc, data):
            observed.update(width=width, height=height, format=format, alloc=alloc)
            np.testing.assert_array_equal(data[0, 0], [11, 79, 203])
            self.assertEqual(data.shape, (4, 6, 3))
            return SimpleNamespace(to_rgb565=lambda: result)

        module = SimpleNamespace(Image=packed_image, RGB888=3, ALLOC_REF=9)
        with patch.dict(sys.modules, image=module):
            self.assertIs(preview_rgb565(frame), result)
        self.assertEqual(observed, dict(width=6, height=4, format=3, alloc=9))

    def test_wrong_camera_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CHW'):
            interleave_rgb(np.zeros((4, 6, 3), dtype=np.uint8), 4, 6)


if __name__ == '__main__':
    unittest.main()
