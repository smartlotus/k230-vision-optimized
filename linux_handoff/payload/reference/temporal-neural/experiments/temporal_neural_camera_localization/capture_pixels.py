"""Reproduce the archived K230 AI2D resize for the fixed 640×360 RGB input."""
import numpy as np


def ai2d_640(planar):
    if planar.shape != (3, 360, 640) or planar.dtype != np.uint8:
        raise ValueError('AI2D replay requires a 640×360 uint8 RGB intermediate')
    # At scale 4, half-pixel centers land exactly midway through the inner
    # four pixels of each 4×4 block. Accumulate before rounding to uint8.
    total = planar[:, 1::4, 1::4].astype(np.uint16)
    total += planar[:, 1::4, 2::4]
    total += planar[:, 2::4, 1::4]
    total += planar[:, 2::4, 2::4]
    result = np.full((3, 96, 160), 114, dtype=np.uint8)
    result[:, 3:-3] = (total + 2) // 4
    return result
