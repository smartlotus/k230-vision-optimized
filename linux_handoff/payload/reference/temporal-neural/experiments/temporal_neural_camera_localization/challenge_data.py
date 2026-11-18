"""Freeze unseen path segments and paired synthetic illumination before model scoring."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image


PILOT_SHA = 'ec2e617bb3b15df10c80f666aa2166f29d7231741026dc187ddc582887609042'
MASK_SHA = 'fa5f526d741e27af830cc4b5a6c3e945ef48442670db3aabd12bd642def6f8d2'
GEOMETRY_SHA = 'b5bea8c6738bc7ebd8d57cdae34af99d1cf10b3fc782ea9c6c986e8802441e95'
DISTANCE_MM = 100.
TARGETS = ('camera_x_mm', 'camera_y_mm')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def polyline(points, count):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.r_[0., np.cumsum(lengths)]
    samples = np.linspace(0, distances[-1], count)
    return np.stack([np.interp(samples, distances, values) for values in points.T], axis=1)


def distance_to_paths(points, paths):
    starts, ends = np.concatenate([(line[:-1], line[1:]) for line in paths], axis=1)
    direction = ends - starts
    distances2 = (direction**2).sum(axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    progress = np.divide((relative * direction).sum(axis=-1), distances2,
                         out=np.zeros(relative.shape[:2]), where=distances2 > 0).clip(0, 1)
    return np.linalg.norm(relative - progress[..., None] * direction, axis=-1).min(axis=1)


def road_only(points, mask):
    col = ((.5 + points[:, 0] / 3000.) * mask.shape[1]).astype(int)
    row = ((.5 - points[:, 1] / 3000.) * mask.shape[0]).astype(int)
    return bool((np.abs(points) <= 1200).all() and np.isin(mask[row, col], [0, 3]).all())


def candidates(paths, mask):
    for outer, cross, family in itertools.product(
            (710, 745, 780, 810, 850, 925, 980, 1030, 1080, 1110, 1140),
            (-165, -100, 80, 165), range(5)):
        a, b = (1050, 1030), (1040, -1060)
        waypoints = {
            0: [a, (outer, outer), (outer, -outer), b],
            1: [a, (outer, outer), (cross, outer), (cross, -outer), (outer, -outer), b],
            2: [a, (outer, outer), (-outer, outer), (-outer, -outer), (outer, -outer), b],
            3: [a, (outer, outer), (outer, cross), (-outer, cross), (-outer, -outer), (outer, -outer), b],
            4: [a, (outer, outer), (-outer, outer), (-outer, cross), (outer, cross), (outer, -outer), b],
        }[family]
        dense = polyline(waypoints, 1500)
        points = polyline(waypoints, 25)
        # The rendered sequence also has to remain on the gray or blue surface
        # along straight segments between its actual sampled positions.
        total = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
        if not road_only(dense, mask) or not road_only(polyline(points, 1500), mask):
            continue
        distances = distance_to_paths(points, paths)
        unseen = distances >= DISTANCE_MM
        whole_window = sum(unseen[i - 7:i + 1].all() for i in range(7, len(points)))
        endpoint = int(unseen[7:].sum())
        yield dict(points=points.tolist(), distances=distances.tolist(), family=family, outer=outer,
                   cross=cross, length_mm=float(total), unseen_warmed=endpoint, unseen_whole_window=int(whole_window))


def plan(pilot, field, output):
    if digest(pilot) != PILOT_SHA or digest(field) != MASK_SHA:
        raise ValueError('Challenge must be frozen against the exact pilot and corrected map')
    groups = defaultdict(list)
    for line in pilot.read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        if row['split'] == 'train':
            groups[row['trajectory_id']].append([row['camera_xy_target'][key] for key in TARGETS])
    if len(groups) != 20 or any(len(line) != 25 for line in groups.values()):
        raise ValueError('Expected the complete original training trajectory set')
    with Image.open(field) as image:
        mask = np.asarray(image)
    pool = list(candidates(list(groups.values()), mask))
    most, near = [], []
    for family in (1, 2, 3, 4):
        available = [route for route in pool if route['family'] == family]
        most.append(max(available, key=lambda route: (route['unseen_whole_window'], route['unseen_warmed'])))
        near.append(min(available, key=lambda route: (route['unseen_warmed'], np.mean(route['distances']))))
    trajectories = []
    for group, routes in (('spatial', most), ('appearance', near)):
        for i, route in enumerate(routes):
            rng = random.Random(487600 + i + 10 * (group == 'appearance'))
            direction = 1 + i % 2
            points = route['points'][::(-1 if direction == 2 else 1)]
            distances = route['distances'][::(-1 if direction == 2 else 1)]
            schedule = []
            period = route['length_mm'] / (24 * rng.uniform(450, 500))
            yaw = rng.uniform(-180, 180)
            for j, (xy, distance) in enumerate(zip(points, distances)):
                schedule.append(dict(xy=xy, distance_to_original_train_path_mm=distance, timestamp_s=j * period,
                                     yaw_deg=yaw + 25 * math.sin(math.tau * j / 24)))
            trajectories.append(dict(id=f'{group}-{i + 1}', group=group, family=route['family'],
                                     start_zone_id=direction, finish_zone_id=3 - direction,
                                     seed=487600 + i + 10 * (group == 'appearance'),
                                     height_mm=rng.uniform(380, 420), diagonal_fov_deg=rng.uniform(92, 100),
                                     pitch_deg=rng.uniform(-1, 1), roll_deg=rng.uniform(-1, 1),
                                     variants=['nominal'] if group == 'spatial' else ['nominal', 'cool' if i % 2 else 'warm'],
                                     frames=schedule))
    value = dict(status='NEURAL_CHALLENGE_FROZEN_BEFORE_SCORING', pilot_manifest_sha256=PILOT_SHA,
                 field_mask_sha256=MASK_SHA, geometry_sha256=GEOMETRY_SHA, unseen_distance_mm=DISTANCE_MM,
                 scored_trajectory_partitions={'spatial': 4, 'appearance_paired': 4}, source_sha256=digest(Path(__file__)),
                 trajectories=trajectories)
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    for name, routes in (('spatial', most), ('appearance', near)):
        print(name, 'unseen_warm_by_route', [r['unseen_warmed'] for r in routes],
              'whole_unseen_windows_by_route', [r['unseen_whole_window'] for r in routes])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('pilot', 'field', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plan(args.pilot, args.field, args.out)
