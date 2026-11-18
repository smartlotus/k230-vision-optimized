"""Audit opposite-blue-zone ambiguities using only synthetic ground-plane semantics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_neural import digest


def sample(mask, world):
    col = np.floor((.5 + world[..., 0] / 3.) * mask.shape[1]).astype(int)
    row = np.floor((.5 - world[..., 1] / 3.) * mask.shape[0]).astype(int)
    valid = (col >= 0) & (row >= 0) & (col < mask.shape[1]) & (row < mask.shape[0])
    return np.where(valid, mask[row.clip(0, mask.shape[0] - 1), col.clip(0, mask.shape[1] - 1)], 255)


def interior(values):
    center = values[1:-1, 1:-1]
    return ((center == values[:-2, 1:-1]) & (center == values[2:, 1:-1])
            & (center == values[1:-1, :-2]) & (center == values[1:-1, 2:]))


def measure(row, field, image):
    meta = row['metadata']
    pose = np.asarray(meta['camera_to_world_blender'])
    intrinsics = meta['intrinsics_pinhole']
    u, v = np.meshgrid(np.arange(intrinsics['width']), np.arange(intrinsics['height']))
    rays = np.stack(((u - intrinsics['cx']) / intrinsics['fx'],
                     (intrinsics['cy'] - v) / intrinsics['fy'], np.full_like(u, -1)), axis=-1)
    direction = rays @ pose[:3, :3].T
    if (direction[..., 2] >= 0).any():
        raise ValueError('This audit expects a downward planar synthetic camera')
    points = pose[:2, 3] - pose[2, 3] / direction[..., 2, None] * direction[..., :2]
    sign = 1 if meta['start_zone_id'] == 1 else -1
    relative = points - [1.2, 1.2 * sign]
    twin = np.stack((sign * relative[..., 1], -sign * relative[..., 0]), axis=-1) + [1.2, -1.2 * sign]
    own, other = sample(field, points), sample(field, twin)
    stable = interior(own) & interior(other) & interior(image)
    if not stable.any():
        raise ValueError('No stable common semantic interior')
    a, b, rendered = [values[1:-1, 1:-1][stable] for values in (own, other, image)]
    return dict(sample_id=row['sample_id'], timestamp_s=row['timestamp_s'], frame_index=row['frame_index'],
                rendered_ground_plane_agreement=float((a == rendered).mean()),
                opposite_blue_zone_difference_fraction=float((a != b).mean()))


def run(data, field, out):
    qa = json.loads((data / 'QA.json').read_text())
    if digest(field) != qa['field_mask_sha256'] or digest(data / 'manifest.jsonl') != qa['manifest_sha256']:
        raise ValueError('Audit field or observations differ')
    with Image.open(field) as image:
        mask = np.asarray(image)
    grouped = defaultdict(list)
    for line in (data / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        grouped[row['trajectory_id']].append(row)
    records = []
    for key, rows in grouped.items():
        evaluated = []
        for row in rows[:16]:
            path = data / row['mask_path']
            if digest(path) != row['mask_sha256']:
                raise ValueError('Semantic frame differs')
            with Image.open(path) as image:
                evaluated.append(measure(row, mask, np.asarray(image)))
        first_different = next((frame['frame_index'] for frame in evaluated
                                if frame['opposite_blue_zone_difference_fraction'] >= .01), None)
        records.append(dict(trajectory=key, first_1pct_difference_frame=first_different, frames=evaluated))
    report = dict(status='PLANAR_SYNTHETIC_OBSERVABILITY_AUDIT_COMPLETE',
                  scope='Ground-plane classes only. This metric does not establish RGB indistinguishability.',
                  frames_measured=sum(len(row['frames']) for row in records), trajectories=len(records),
                  starts_with_zero_planar_class_difference=sum(row['frames'][0]['opposite_blue_zone_difference_fraction'] == 0 for row in records),
                  starts_with_99pct_rendered_ground_match=sum(row['frames'][0]['rendered_ground_plane_agreement'] >= .99 for row in records),
                  first16_below_1pct_difference=sum(row['first_1pct_difference_frame'] is None for row in records),
                  map_sha256=digest(field), manifest_sha256=qa['manifest_sha256'],
                  source_sha256=digest(Path(__file__)), records=records, real_camera='NOT_MEASURED')
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'field', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.field, args.out)
