"""Evaluate frozen models on the geometry pilot's locally withheld test split."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from challenge_data import PILOT_SHA
from cloud_geometry import sha, training_manifest
from evaluate_neural import SETTINGS, TARGETS, clip, digest, load_sequences, model_manifest_identity, read, score_grid, statistics, verified_graph
from geometry_pilot import SEMANTIC_CLASSES, VISIBLE_FRACTION, distance_to_paths


def evaluate(sequences, spatial_ids, visible, model, lineage, settings, common_window=8):
    collection = read(model / 'MODEL_COLLECTION.json')
    if collection['status'] != 'ORIGINAL_NAMED_MODEL_MEMBERS_SHA256_PASS':
        raise ValueError('Expected verified exact-version Kaggle member collection')
    output = {}
    for setting in settings:
        session, name, centers, metrics, _, _ = verified_graph(model, collection, setting, lineage)
        _, window = SETTINGS[setting]
        modes = ('chronological', 'current_only') if window > 1 else ('chronological',)
        results = {}
        for mode in modes:
            slices = {}
            span = []
            initial = {'correct': 0, 'wrong_blue_zone': 0, 'unlocalized_outside_blue_zones': 0}
            for rows, frames in sequences:
                previous_xy = None
                for i, row in enumerate(rows):
                    value = clip(frames, i, window, mode)
                    raw, = session.run(['location_logits'], {name: value})
                    if not np.isfinite(raw).all():
                        raise ValueError('Nonfinite geometry pilot prediction')
                    target = np.asarray([row['camera_xy_target'][key] for key in TARGETS])
                    xy, error, confidence = score_grid(raw[0], centers, target, metrics['temperature'], 48)
                    jump = float(np.linalg.norm(xy - previous_xy)) if previous_xy is not None else 0.
                    previous_xy = xy
                    if i == 0:
                        zone = (1 if xy[1] >= 900 else 2) if xy[0] >= 900 and abs(xy[1]) >= 900 else None
                        if zone is None:
                            initial['unlocalized_outside_blue_zones'] += 1
                        elif zone == row['metadata']['start_zone_id']:
                            initial['correct'] += 1
                        else:
                            initial['wrong_blue_zone'] += 1
                    tags = ['all_unseen_trajectories', row['challenge']]
                    if i >= common_window - 1:
                        tags.append('all_complete_windows')
                        span.append(row['timestamp_s'] - rows[i - common_window + 1]['timestamp_s'])
                        window_visibility = [visible[frame['sample_id']] for frame in rows[i - common_window + 1:i + 1]]
                        if window_visibility[-1]:
                            tags.append('landmark_visible_in_current_frame')
                        elif any(window_visibility[:-1]):
                            tags.append('landmark_visible_only_in_past')
                        else:
                            tags.append('landmark_sparse_entire_window')
                    if row['sample_id'] in spatial_ids:
                        tags.append('all_8_at_least_100mm_from_revised_train')
                    if row['challenge'] == 'appearance':
                        tags.append(row['challenge'] + '_' + row['appearance'])
                    result = dict(error_mm=error, confidence=confidence, jump_mm=jump)
                    for tag in tags:
                        slices.setdefault(tag, []).append(result)
            results[mode] = dict(metrics={tag: statistics(values) for tag, values in slices.items()},
                                 common_window=common_window, common_window_span_median_s=float(np.median(span)),
                                 first_frame_blue_zone_top1=initial)
        output[setting] = dict(selected_epoch=metrics['selected_epoch'], optimizer_updates=metrics['optimizer_updates'],
                               float_onnx_sha256=metrics['onnx_sha256'], best_sha256=metrics['best_sha256'],
                               settings=results)
    return dict(source=collection['source'], training_manifest_sha256=lineage, models=output)


def run(data, audit, cloud, baseline, candidate, output, settings=('cnn1', 'tcn8')):
    qa = read(audit)
    frozen = read(cloud / 'SUBMISSION_INTENT.json')
    cloud_result = read(cloud / 'RESULT.json')
    if (qa['status'] != 'GEOMETRY_PILOT_AUDIT_PASS' or frozen['input']['audit_sha256'] != digest(audit)
            or cloud_result['run_id'] != frozen['run_id'] or cloud_result['compute_status'] != 'PASS'):
        raise ValueError('Geometry audit or cloud output differs from the frozen run')
    subset, _ = training_manifest(data / 'manifest.jsonl')
    if sha(subset) != frozen['input']['training_manifest_sha256'] or sha(subset) != cloud_result['dataset_manifest_sha256']:
        raise ValueError('Local train/validation subset differs from exact cloud inputs')
    torch.set_num_threads(2)
    sequences, manifest, field_sha = load_sequences(data)
    if manifest != qa['manifest_sha256'] or manifest != frozen['input']['full_local_manifest_sha256']:
        raise ValueError('Expected the audited, locally withheld test trajectories')
    n = sum(len(rows) for rows, _ in sequences)
    if n != 250 or len(sequences) != 10 or qa['spatial_unseen_all_8'] != len(set(qa['spatial_sample_ids'])):
        raise ValueError('Geometry pilot test coverage differs')
    spatial_ids = set(qa['spatial_sample_ids'])
    lines = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines()]
    train_paths = {}
    for row in lines:
        if row['split'] == 'train':
            train_paths.setdefault(row['trajectory_id'], []).append([row['camera_xy_target'][key] for key in TARGETS])
    spatial_coverage = {}
    for setting in settings:
        window = SETTINGS[setting][1]
        count = 0
        for rows, _ in sequences:
            points = np.array([[r['camera_xy_target'][key] for key in TARGETS] for r in rows])
            distances = distance_to_paths(points, [np.asarray(path) for path in train_paths.values()])
            count += sum(distances[i - window + 1:i + 1].min() >= 100 for i in range(window - 1, len(rows)))
        spatial_coverage[setting] = dict(window_frames=window, full_history_at_least100mm_from_train=int(count))
    visible = {}
    for rows, _ in sequences:
        for row in rows:
            path = (data / row['mask_path']).resolve()
            if not path.is_relative_to(data.resolve()) or digest(path) != row['mask_sha256']:
                raise ValueError('Withheld semantic mask differs from its manifest')
            with Image.open(path) as im:
                visible[row['sample_id']] = np.isin(np.asarray(im), SEMANTIC_CLASSES).mean() >= VISIBLE_FRACTION
    common_window = max(8, max(SETTINGS[setting][1] for setting in settings))
    geometry = evaluate(sequences, spatial_ids, visible, candidate, sha(subset), settings, common_window)
    baseline_sha = torch.load(baseline / 'tcn8/best.pt', weights_only=True, map_location='cpu')['dataset_manifest_sha256']
    if baseline_sha != PILOT_SHA:
        baseline_sha = model_manifest_identity(data, baseline, 'tcn8', manifest)
    earlier = evaluate(sequences, spatial_ids, visible, baseline, baseline_sha, ('cnn1', 'tcn8'), common_window)
    reference = geometry['models'][settings[-1]]['settings']['chronological']['metrics']
    gates = {name: reference[name]['p95_mm'] <= 500 for name in (
        'all_unseen_trajectories', 'all_8_at_least_100mm_from_revised_train', 'appearance')}
    value = dict(status='NEURAL_GEOMETRY_PILOT_EVAL_COMPLETE', test_manifest_sha256=manifest, test_frames=n,
                 test_trajectories=len(sequences), mask_sha256=field_sha, older_appearance=earlier,
                 trained_geometry=geometry, spatial_definition='Distance to revised pilot train polylines only',
                 full_model_window_spatial_coverage=spatial_coverage,
                 candidate_p95_500mm_gates=gates, quality='PASS_GEOMETRY_PILOT' if all(gates.values()) else 'HOLD',
                 hardware='NOT_RUN', source_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    for name, group in (('prior', earlier), ('geometry', geometry)):
        for key, model in group['models'].items():
            print(name, key, {k: (m['frames'], round(m['median_mm']), round(m['p95_mm']))
                             for k, m in model['settings']['chronological']['metrics'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'audit', 'cloud', 'baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=['cnn1', 'tcn8'])
    args = parser.parse_args()
    run(args.data, args.audit, args.cloud, args.baseline, args.candidate, args.out, args.settings)
