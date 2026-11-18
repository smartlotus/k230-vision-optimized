"""Freeze long western trajectories with complete spatially unseen 16-frame histories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from challenge_data import distance_to_paths, road_only
from geometry_pilot import route_points


def plan(data, geometry_plan, field, out):
    base = json.loads(geometry_plan.read_text(encoding='utf8'))
    rows = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            groups[row['trajectory_id']].append([row['camera_xy_target'][key]
                                                  for key in ('camera_x_mm', 'camera_y_mm')])
    training = [np.asarray(points) for points in groups.values()]
    with Image.open(field) as image:
        mask = np.asarray(image)
    trajectories = []
    for index in range(6):
        rng = random.Random(604000 + index)
        for _ in range(100):
            route = route_points(rng, 2, west=True)
            points = [route[0]]
            for a, b, n in zip(route, route[1:], (2, 3, 2, 25, 2, 3, 2)):
                points.extend(np.linspace(a, b, n + 1)[1:])
            xy = np.asarray(points)
            if all(road_only(np.linspace(a, b, 100), mask) for a, b in zip(xy, xy[1:])):
                break
        else:
            raise ValueError('Could not build a continuous western challenge route')
        if index % 2:
            xy = xy[::-1]
        distances = distance_to_paths(xy, training)
        count = int(sum(distances[i - 15:i + 1].min() >= 100 for i in range(15, 40)))
        if len(xy) != 40 or count < 10:
            raise ValueError('Expected at least ten complete spatially unseen long windows')
        intervals = np.maximum(.23, np.linalg.norm(np.diff(xy, axis=0), axis=1) / 650)
        timestamps = np.r_[0., np.cumsum(intervals)]
        yaw = rng.uniform(-180, 180)
        frames = [dict(xy=point.tolist(), timestamp_s=float(t), yaw_deg=float(yaw + 20 * np.sin(2 * np.pi * i / 39)),
                       distance_to_original_train_path_mm=float(d))
                  for i, (point, t, d) in enumerate(zip(xy, timestamps, distances))]
        trajectories.append(dict(id=f'long-west-{index}', group='spatial', split='test', seed=604000 + index,
                                 start_zone_id=index % 2 + 1, finish_zone_id=2 - index % 2, variants=['nominal'],
                                 height_mm=rng.uniform(385, 415), diagonal_fov_deg=rng.uniform(93, 99),
                                 pitch_deg=rng.uniform(-1, 1), roll_deg=rng.uniform(-1, 1),
                                 full_unseen16=count, frames=frames))
    base.update(status='NEURAL_CHALLENGE_FROZEN_BEFORE_SCORING', expected_frames=240, trajectories=trajectories,
                purpose='Every scored full16 history stays at least100mm from revised train polylines',
                source_geometry_manifest_sha256='90e05cacc96118ff7ec35871bec0fba5e5458799c5f620a3ad3bb84d9b8fe0df')
    out.write_text(json.dumps(base, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Frozen new frames', 240, 'whole spatially unseen T16 endpoints',
          sum(tr['full_unseen16'] for tr in trajectories), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'plan', 'field', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plan(args.data, args.plan, args.field, args.out)
