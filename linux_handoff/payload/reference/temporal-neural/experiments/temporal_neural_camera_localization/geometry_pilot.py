"""Plan a landmark-aware pilot with continuous spatially separated test windows."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

from challenge_data import DISTANCE_MM, GEOMETRY_SHA, MASK_SHA, PILOT_SHA, TARGETS, digest, distance_to_paths, road_only


LANDMARKS = np.asarray([(0, 1190), (-1125, 0), (0, -1125), (1190, 0)], dtype=float)
SEMANTIC_CLASSES = (2, 6)
VISIBLE_FRACTION = .01
WINDOW = 8
FRAMES = 25


def route_points(rng, family, west=False):
    a = (rng.uniform(985, 1105), rng.uniform(980, 1085))
    b = (rng.uniform(985, 1105), rng.uniform(-1095, -980))
    side = rng.uniform(790, 880)
    edge = rng.uniform(935, 1020)
    lane = rng.uniform(-143, 143)
    if west:
        # The white plate stops at x = -1050 mm. Keep the held-out lane
        # between it and the leftmost training corridor.
        west_x = rng.uniform(-1030, -1010)
        return [a, (side, edge), (west_x, edge), (west_x, 700), (west_x, -700),
                (west_x, -edge), (side, -edge), b]
    if family == 0:
        return [a, (side, edge), (side, -edge), b]
    if family == 1:
        return [a, (side, edge), (lane, edge), (lane, -edge), (side, -edge), b]
    if family == 2:
        left = rng.uniform(-860, -800)
        return [a, (side, edge), (left, edge), (left, -edge), (side, -edge), b]
    if family == 3:
        return [a, (side, edge), (side, lane), (lane, lane), (lane, -edge), (side, -edge), b]
    return [a, (side, edge), (lane, edge), (lane, lane), (side, lane), (side, -edge), b]


def resample(points, favor_landmarks, west=False):
    points = np.asarray(points)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    nominal = np.linspace(0, 1, 64)
    previews = points[:-1, None, :] + nominal[None, :, None] * delta[:, None, :]
    d = np.linalg.norm(previews[:, :, None] - LANDMARKS[None, None, ...], axis=3).min(axis=2)
    weights = 1 + 2 * np.exp(-.5 * (d / 280.)**2) if favor_landmarks else np.ones_like(d)
    mass = weights.mean(axis=1) * lengths
    steps = np.ones(len(lengths), dtype=int)
    if west:
        steps[3] = 10
    for _ in range(FRAMES - 1 - steps.sum()):
        steps[int((mass / (steps + .5)).argmax())] += 1
    result = [points[0]]
    for a, b, weight, count in zip(points, points[1:], weights, steps):
        cumulative = np.r_[0, np.cumsum((weight[:-1] + weight[1:]) / 2)]
        positions = np.interp(np.linspace(0., cumulative[-1], count + 1)[1:], cumulative, nominal)
        result.extend(a + positions[:, None] * (b - a))
    return np.asarray(result)


def create_plan(mask_path, out):
    if digest(mask_path) != MASK_SHA:
        raise ValueError('Expected the audited corrected field mask')
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    trajectories, training_paths, training_seeds = [], [], set()
    groups = [('train', 'landmark_train', 20), ('val', 'validation', 10),
              ('test', 'spatial_west', 6), ('test', 'appearance', 4)]
    for split, group, count in groups:
        for index in range(count):
            seed = 523000 + 100 * len(trajectories) + index
            rng = random.Random(seed)
            family = index // 2 % 5
            start = 1 + index % 2
            west = group == 'spatial_west'
            for _ in range(100):
                xy = resample(route_points(rng, family, west), split == 'train', west)
                if all(road_only(np.linspace(a, b, 100), mask) for a, b in zip(xy, xy[1:])):
                    break
            else:
                raise ValueError('Could not build a continuous corrected-field route')
            if start == 2:
                xy = xy[::-1]
            if split == 'train':
                training_paths.append(xy)
                training_seeds.add(seed)
            distance = distance_to_paths(xy, training_paths)
            if west and sum(np.min(distance[i - WINDOW + 1:i + 1]) >= DISTANCE_MM
                            for i in range(WINDOW - 1, len(xy))) < 4:
                raise ValueError('Expected at least four spatially separated warmed windows on the west lane')
            if split != 'train' and seed in training_seeds:
                raise ValueError('Trajectory seed crosses the split')
            if group == 'appearance':
                style = ('warm', 'cool')[index // 2]
            elif split == 'train':
                style = ('nominal', 'nominal', 'warm', 'cool')[index // 2 % 4]
            else:
                style = 'nominal'
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            dt = np.maximum(.23, steps / rng.uniform(570, 730))
            times = np.r_[0., np.cumsum(dt)]
            yaw = rng.uniform(-180, 180)
            frames = [dict(xy=point.tolist(), timestamp_s=float(t),
                           yaw_deg=yaw + 25 * math.sin(math.tau * i / 24),
                           distance_to_original_train_path_mm=float(d))
                      for i, (point, t, d) in enumerate(zip(xy, times, distance))]
            trajectories.append(dict(id=f'geo-{group}-{index:02}', group=group, split=split, seed=seed,
                                     family=family, start_zone_id=start, finish_zone_id=3 - start,
                                     height_mm=rng.uniform(380, 420), diagonal_fov_deg=rng.uniform(92, 100),
                                     pitch_deg=rng.uniform(-1, 1), roll_deg=rng.uniform(-1, 1),
                                     variants=[style], frames=frames))
    value = dict(status='GEOMETRY_PILOT_FROZEN_BEFORE_SCORING', pilot_manifest_sha256=PILOT_SHA,
                 field_mask_sha256=MASK_SHA, geometry_sha256=GEOMETRY_SHA, unseen_distance_mm=DISTANCE_MM,
                 expected_frames=1000, expected_split_frames=dict(train=500, val=250, test=250),
                 source_sha256=digest(Path(__file__)), trajectories=trajectories)
    out.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')


def audit(data, plan_path, field_path, baseline, out):
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    qa = json.loads((data / 'QA.json').read_text(encoding='utf8'))
    manifest = data / 'manifest.jsonl'
    if (qa['manifest_sha256'] != digest(manifest) or qa['plan_sha256'] != digest(plan_path)
            or qa['field_mask_sha256'] != digest(field_path) or digest(baseline / 'manifest.jsonl') != PILOT_SHA):
        raise ValueError('Geometry pilot inputs differ from their identities')
    prior = [json.loads(line) for line in (baseline / 'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    if set(r['image_sha256'] for r in rows) & set(r['image_sha256'] for r in prior):
        raise ValueError('An RGB observation exactly duplicates the original pilot')
    if Counter(row['split'] for row in rows) != plan['expected_split_frames']:
        raise ValueError('Geometry pilot split counts differ')
    ids = set()
    groups = defaultdict(list)
    for row in rows:
        if row['sample_id'] in ids or set(row['camera_xy_target']) != set(TARGETS):
            raise ValueError('Expected unique geometry pilot IDs and XY-only targets')
        ids.add(row['sample_id'])
        for field in ('image', 'mask'):
            if digest(data / row[field + '_path']) != row[field + '_sha256']:
                raise ValueError('Geometry pilot member has changed')
        groups[row['pose_group_id']].append(row)
    if len(groups) != 40 or any(len(sequence) != FRAMES for sequence in groups.values()):
        raise ValueError('Expected forty complete independent trajectories')
    for key in ('trajectory_id', 'appearance_profile_id', 'pose_group_id'):
        assignment = defaultdict(set)
        for row in rows:
            assignment[row[key]].add(row['split'])
        if any(len(splits) > 1 for splits in assignment.values()):
            raise ValueError('Split leakage in ' + key)

    def xy(sequence):
        return np.array([[row['camera_xy_target'][key] for key in TARGETS] for row in sequence])

    train = [xy(seq) for seq in groups.values() if seq[0]['split'] == 'train']
    old_train = [row for row in prior if row['split'] == 'train']
    new_train = [row for row in rows if row['split'] == 'train']
    with Image.open(field_path) as im:
        field = np.asarray(im)
    profiles = Counter()
    unseen, unseen_by_route = [], {}
    max_speed = 0.
    for seq in groups.values():
        points = xy(seq)
        ts = np.array([r['timestamp_s'] for r in seq])
        maximum = float((np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(ts)).max())
        max_speed = max(max_speed, maximum)
        if not all(road_only(np.linspace(a, b, 100), field) for a, b in zip(points, points[1:])):
            raise ValueError('A route leaves the gray road and blue zones')
        for point, zone in ((points[0], seq[0]['metadata']['start_zone_id']),
                            (points[-1], seq[0]['metadata']['finish_zone_id'])):
            x, y = point
            if not (900 <= x <= 1200 and 900 <= y * (1 if zone == 1 else -1) <= 1200):
                raise ValueError('Geometry pilot endpoints differ from the two-blue-zone contract')
        if seq[0]['challenge'] == 'spatial_west':
            distance = distance_to_paths(points, train)
            accepted = [seq[i]['sample_id'] for i in range(WINDOW - 1, len(seq))
                        if distance[i - WINDOW + 1:i + 1].min() >= DISTANCE_MM]
            unseen.extend(accepted)
            unseen_by_route[seq[0]['trajectory_id']] = len(accepted)
        profiles[seq[0]['split'] + '/' + seq[0]['challenge'] + '/' + seq[0]['appearance']] += len(seq)

    def coverage(data_root, frames):
        count = 0
        for row in frames:
            path = data_root / row['mask_path']
            if digest(path) != row['mask_sha256']:
                raise ValueError('Semantic mask differs from its source manifest')
            with Image.open(path) as image:
                count += float(np.isin(np.asarray(image), SEMANTIC_CLASSES).mean()) >= VISIBLE_FRACTION
        return dict(total=len(frames), visible=count, fraction=count / len(frames))

    geometry = dict(status='GEOMETRY_PILOT_AUDIT_PASS', manifest_sha256=digest(manifest),
                    plan_sha256=digest(plan_path), corrected_field_mask_sha256=qa['field_mask_sha256'],
                    frames=len(rows), trajectories=len(groups), split_profiles=dict(profiles),
                    train_fixed_landmark_visibility=dict(old=coverage(baseline, old_train), revised=coverage(data, new_train)),
                    landmark_contract=dict(classes=SEMANTIC_CLASSES, minimum_area_fraction=VISIBLE_FRACTION),
                    spatial_warm_cutoff_mm=DISTANCE_MM, spatial_unseen_all_8=len(unseen),
                    spatial_windows_by_trajectory=unseen_by_route, spatial_sample_ids=unseen, maximum_speed_sim_mm_s=max_speed)
    if (geometry['train_fixed_landmark_visibility']['revised']['visible']
            <= geometry['train_fixed_landmark_visibility']['old']['visible'] or len(unseen) < 24):
        raise ValueError('Pilot must increase landmark-visible training frames and wholly distant test windows')
    out.write_text(json.dumps(geometry, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({k: v for k, v in geometry.items() if k not in ('spatial_sample_ids',)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('plan')
    check = sub.add_parser('audit')
    for command, names in ((build, ('field', 'out')), (check, ('data', 'plan', 'field', 'baseline', 'out'))):
        for name in names:
            command.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        create_plan(args.field, args.out)
    else:
        audit(args.data, args.plan, args.field, args.baseline, args.out)
