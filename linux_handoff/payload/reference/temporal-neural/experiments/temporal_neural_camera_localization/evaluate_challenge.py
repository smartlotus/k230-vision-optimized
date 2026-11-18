"""Score unchanged neural models on a fresh spatial and appearance challenge."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from challenge_data import DISTANCE_MM, PILOT_SHA, road_only
from evaluate_neural import SETTINGS, TARGETS, clip, digest, load_sequences, read, score_grid, statistics, verified_graph


def verify(data, plan_path, pilot, field):
    identity = read(data / 'QA.json')
    if (identity['status'] != 'RENDER_COMPLETE' or digest(plan_path) != identity['plan_sha256']
            or digest(field) != identity['field_mask_sha256']
            or not digest(pilot) == identity['original_pilot_manifest_sha256'] == PILOT_SHA):
        raise ValueError('Challenge must match the frozen schedule, field and training manifest')
    plan = read(plan_path)
    if (identity['geometry_sha256'] != plan['geometry_sha256']
            or digest(field) != plan['field_mask_sha256'] or plan['pilot_manifest_sha256'] != PILOT_SHA):
        raise ValueError('Challenge plan identity differs')
    sequences, challenge_sha, _ = load_sequences(data)
    expected = {tr['id'] + '-' + style: (tr, style) for tr in plan['trajectories'] for style in tr['variants']}
    if {rows[0]['trajectory_id'] for rows, _ in sequences} != set(expected):
        raise ValueError('Rendered challenge trajectories differ from the frozen plan')
    prior = [json.loads(line) for line in pilot.read_text(encoding='utf8').splitlines()]
    prior_ids = {r['sample_id'] for r in prior}
    prior_rgb = {r['image_sha256'] for r in prior}
    with Image.open(field) as image:
        mask = np.asarray(image)
    by_pose = {}
    partition_frames = {}
    frame_count = 0
    for rows, _ in sequences:
        tr, style = expected[rows[0]['trajectory_id']]
        if len(rows) != len(tr['frames']):
            raise ValueError('Incomplete challenge route')
        positions = []
        for index, (row, frame) in enumerate(zip(rows, tr['frames'])):
            xy = [row['camera_xy_target'][key] for key in TARGETS]
            if (row['sample_id'] in prior_ids or row['image_sha256'] in prior_rgb or row['frame_index'] != index
                    or row['pose_group_id'] != tr['id'] or row['challenge'] != tr['group'] or row['appearance'] != style
                    or row['timestamp_s'] != frame['timestamp_s']
                    or row['distance_to_original_train_path_mm'] != frame['distance_to_original_train_path_mm']
                    or not np.allclose(xy, frame['xy'], atol=.001, rtol=0)):
                raise ValueError('Challenge frame differs from frozen inputs')
            positions.append(xy)
            path = (data / row['mask_path']).resolve()
            if not path.is_relative_to(data.resolve()) or digest(path) != row['mask_sha256']:
                raise ValueError('Render mask was changed')
            with Image.open(path) as image:
                mask_samples = np.asarray(image)
            row['fixed_landmark_area_fraction'] = float(np.isin(mask_samples, (2, 6)).mean())
            pairing = (tr['id'], index)
            by_pose.setdefault(pairing, []).append((row['mask_sha256'], row['image_sha256']))
        for xy, zone in ((positions[0], tr['start_zone_id']), (positions[-1], tr['finish_zone_id'])):
            x, y = xy
            if not (900 <= x <= 1200 and 900 <= y * (1 if zone == 1 else -1) <= 1200):
                raise ValueError('Challenge endpoints must reach the two different blue zones')
        if any(not road_only(np.linspace(a, b, 100), mask) for a, b in zip(positions, positions[1:])):
            raise ValueError('Challenge crossed a non-road surface')
        partition_frames[tr['group']] = partition_frames.get(tr['group'], 0) + len(rows)
        frame_count += len(rows)
    pairs = 0
    for samples in by_pose.values():
        if len(samples) == 2:
            (mask_a, rgb_a), (mask_b, rgb_b) = samples
            if mask_a != mask_b or rgb_a == rgb_b:
                raise ValueError('Paired lighting frames must change RGB while preserving semantics')
            pairs += 1
    if frame_count != identity['frames'] or pairs != 100:
        raise ValueError('Missing rendered challenge observations or paired lighting cases')
    return sequences, challenge_sha, dict(frames=frame_count, sequences=len(sequences), paired_frames=pairs,
                                         partition_frames=partition_frames)


def score(sequences, session, input_name, centers, window, temperature):
    cases = {}
    paired = {}
    for rows, frames in sequences:
        previous = None
        for i, row in enumerate(rows):
            rgb = clip(frames, i, window, 'chronological')
            logits, = session.run(['location_logits'], {input_name: rgb})
            if not np.isfinite(logits).all():
                raise ValueError('Challenge inference produced nonfinite logits')
            target = np.asarray([row['camera_xy_target'][key] for key in TARGETS])
            xy, error, confidence = score_grid(logits[0], centers, target, temperature, 48)
            jump = float(np.linalg.norm(xy - previous)) if previous is not None else 0.
            previous = xy
            record = dict(error_mm=error, confidence=confidence, jump_mm=jump)
            if row['challenge'] == 'spatial':
                tags = ['spatial_all']
                if i >= 7:
                    tags.append('spatial_warm')
                    unseen = [r['distance_to_original_train_path_mm'] >= DISTANCE_MM for r in rows[i - 7:i + 1]]
                    if unseen[-1]:
                        tags.append('spatial_warm_endpoint_unseen')
                    if all(unseen):
                        tags.append('spatial_all_eight_unseen')
                tag = 'visible' if row['fixed_landmark_area_fraction'] >= .01 else 'sparse'
                tags.append('spatial_landmark_' + tag)
            else:
                tags = ['appearance_' + row['appearance']]
                paired.setdefault((row['pose_group_id'], i), {})[row['appearance']] = error
            for tag in tags:
                cases.setdefault(tag, []).append(record)
    changes = []
    for appearances in paired.values():
        baseline = appearances['nominal']
        shifted, = (e for variant, e in appearances.items() if variant != 'nominal')
        changes.append(shifted - baseline)
    return dict(slices={name: statistics(rows) for name, rows in cases.items()},
                paired_lighting_delta_mm=dict(frames=len(changes), median=float(np.median(changes)),
                                               p95=float(np.quantile(changes, .95)),
                                               shifts_increasing_error=sum(delta > 0 for delta in changes)))


def run(data, plan, pilot, field, models, out, settings):
    torch.set_num_threads(2)
    sequences, fresh_sha, qa = verify(data, plan, pilot, field)
    collection = read(models / 'MODEL_COLLECTION.json')
    original = []
    for name in settings:
        session, input_name, centers, model, _, _ = verified_graph(models, collection, name, PILOT_SHA)
        result = score(sequences, session, input_name, centers, SETTINGS[name][1], model['temperature'])
        original.append((name, dict(result, checkpoint_sha256=model['best_sha256'],
                                    float_onnx_sha256=model['onnx_sha256'], temperature=model['temperature'])))
        print(name, {tag: (m['frames'], round(m['p95_mm'])) for tag, m in result['slices'].items()}, flush=True)
    record = dict(status='FROZEN_NEURAL_CHALLENGE_COMPLETE', model_source=collection['source'],
                  original_pilot_manifest_sha256=PILOT_SHA, independent_challenge_manifest_sha256=fresh_sha,
                  plan_sha256=digest(plan), source_sha256=digest(Path(__file__)), qa=qa,
                  slice_contract=dict(unseen_distance_from_continuous_train_polyline_mm=DISTANCE_MM,
                                      spatial_warm_common_minimum_frame_index=7, visible_mask_fraction=.01),
                  settings=dict(original))
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'plan', 'pilot', 'field', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=['cnn1', 'tcn8'])
    args = parser.parse_args()
    run(args.data, args.plan, args.pilot, args.field, args.models, args.out, args.settings)
