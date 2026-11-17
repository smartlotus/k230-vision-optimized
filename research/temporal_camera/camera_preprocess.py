"""RGB input matching the verified CanMV uint8 resize for 640×360 sources."""
import numpy as np
from PIL import Image, ImageOps


def ai2d_downsample_4x(rgb):
    """Half-pixel bilinear at 4:1, rounds halves up, then pads to 96×160."""
    if rgb.shape != (3, 360, 640) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 planar RGB at 640×360')
    total = rgb[:, 1::4, 1::4].astype(np.uint16)
    total += rgb[:, 1::4, 2::4]
    total += rgb[:, 2::4, 1::4]
    total += rgb[:, 2::4, 2::4]
    result = np.full((3, 96, 160), 114, dtype=np.uint8)
    result[:, 3:93, :] = (total + 2) // 4
    return result


def rgb_uint8(path):
    with Image.open(path) as source:
        rgb = source.convert('RGB')
        if rgb.size == (640, 360):
            return ai2d_downsample_4x(np.asarray(rgb).transpose(2, 0, 1))
        rgb = rgb.resize((160, 90), Image.Resampling.BILINEAR)
        return np.asarray(ImageOps.expand(rgb, (0, 3, 0, 3), fill=(114, 114, 114))).transpose(2, 0, 1).copy()
