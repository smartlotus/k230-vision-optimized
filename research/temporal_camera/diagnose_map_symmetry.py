"""Compare static floor projections at the true and rotated simulation locations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import digest


def projected_pixels(row):
    metadata = row['metadata']
    camera = np.asarray(metadata['camera_to_world_blender'])
    calibration = metadata['intrinsics_pinhole']
    u, v = np.meshgrid((np.arange(calibration['width']) - calibration['cx']) / calibration['fx'],
                       (calibration['cy'] - np.arange(calibration['height'])) / calibration['fy'])
    rays = np.stack((u, v, -np.ones_like(u)), axis=2) @ camera[:3, :3].T
    xyz = camera[:3, 3] - camera[2, 3] * rays / rays[..., 2:3]
    return xyz[..., 0], xyz[..., 1]


def agreement(field, x, y, mask, weights):
    h, w = field.shape
    col = np.floor((x / 3 + .5) * w).astype(np.int32)
    row = np.floor((.5 - y / 3) * h).astype(np.int32)
    projection = field[row.clip(0, h - 1), col.clip(0, w - 1)]
    projection[(col < 0) | (row < 0) | (col >= w) | (row >= h)] = 255
    return float(np.sum((projection == mask) * weights) / weights.sum())


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'field', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = args.data / 'manifest.jsonl'
    with Image.open(args.field) as image:
        field = np.array(image)
    levels = np.zeros(256, np.float32)
    levels[[0, 1, 2, 3, 6, 255]] = [1., 2., 2., 1., 2., 1.]
    gaps = defaultdict(list)
    alignment = defaultdict(list)
    start = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        split = row['split']
        if split not in ('val', 'test'):
            continue
        path = args.data / row['mask_path']
        if digest(path) != row['mask_sha256']:
            raise ValueError('Rendered semantic mask differs from the frozen manifest')
        with Image.open(path) as image:
            mask, counts = np.array(image), image.histogram()
        weights = levels[mask]
        x, y = projected_pixels(row)
        target = row['camera_xy_target']
        cx, cy = target['camera_x_mm'], target['camera_y_mm']
        true_score = agreement(field, x, y, mask, weights)
        alignment[split].append(true_score)
        rotations = ((y, -x), (-y, x), (-x, -y))
        guesses = ((cy, -cx), (-cy, cx), (-cx, -cy))
        alternatives = [agreement(field, px, py, mask, weights)
                        for (px, py), (qx, qy) in zip(rotations, guesses)
                        if np.hypot(cx - qx, cy - qy) >= 500]
        if row['frame_index'] == 0:
            twin_x, twin_y = rotations[0 if cy > 0 else 1]
            twin_score = agreement(field, twin_x, twin_y, mask, weights)
            start.append(dict(sample_id=row['sample_id'], split=split, true_score=true_score,
                              opposite_start_score=twin_score))
        if alternatives:
            gap = true_score - max(alternatives)
            gaps[split + '/all'].append(gap)
            if any(counts[code] >= 15 for code in (1, 2, 6)):
                gaps[split + '/visible_mapped_mark'].append(gap)
    result = {
        'status': 'SIMULATION_ONLY_STATIC_MASK_DIAGNOSTIC',
        'manifest_sha256': digest(manifest), 'field_mask_sha256': digest(args.field),
        'scope': 'True camera projection for diagnosis only, no labels or metadata used for model training or inference',
        'geometry_agreement_median': {s: float(np.median(values)) for s, values in alignment.items()},
        'startup_exact_ties': sum(r['true_score'] == r['opposite_start_score'] for r in start),
        'startup_frames': len(start), 'startup': start,
        'far_rotation_gap': {k: {'frames': len(v), 'alternative_within_2pp_or_better': sum(bool(g <= .02) for g in v)}
                             for k, v in gaps.items()},
        'native_k230': 'NOT RUN',
    }
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'startup'}))


if __name__ == '__main__':
    main()
