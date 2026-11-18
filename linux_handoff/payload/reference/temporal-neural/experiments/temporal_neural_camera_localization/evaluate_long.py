"""Compare the 8/16-frame neural models on the same wholly unseen long histories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from challenge_data import distance_to_paths, road_only
from evaluate_geometry import evaluate
from evaluate_neural import TARGETS, digest, load_sequences, model_manifest_identity, read


def run(data, plan_path, reference, field, baseline, candidate, out, setting='tcn16'):
    torch.set_num_threads(2)
    qa = read(data / 'QA.json')
    plan = read(plan_path)
    sequences, test_sha, _ = load_sequences(data)
    reference_sha = digest(reference / 'manifest.jsonl')
    if (qa['plan_sha256'] != digest(plan_path) or qa['manifest_sha256'] != test_sha
            or plan['source_geometry_manifest_sha256'] != reference_sha or digest(field) != qa['field_mask_sha256']
            or qa['frames'] != 240 or len(sequences) != 6 or sum(len(seq) for seq, _ in sequences) != 240):
        raise ValueError('Expected the frozen six-route independent long-window challenge')
    expected = {item['id'] + '-nominal': item for item in plan['trajectories']}
    training = defaultdict(list)
    for line in (reference / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        if row['split'] == 'train':
            training[row['trajectory_id']].append([row['camera_xy_target'][key] for key in TARGETS])
    train_paths = [np.asarray(path) for path in training.values()]
    with Image.open(field) as image:
        mask = np.asarray(image)
    selected, visible = [], {}
    for rows, _ in sequences:
        tr = expected[rows[0]['trajectory_id']]
        if len(rows) != 40:
            raise ValueError('Expected forty frames per challenge trajectory')
        points = np.array([[row['camera_xy_target'][key] for key in TARGETS] for row in rows])
        if not all(road_only(np.linspace(a, b, 100), mask) for a, b in zip(points, points[1:])):
            raise ValueError('Long challenge left the gray road or blue zones')
        distances = distance_to_paths(points, train_paths)
        ids = [rows[i]['sample_id'] for i in range(15, 40) if distances[i - 15:i + 1].min() >= 100]
        if len(ids) != tr['full_unseen16']:
            raise ValueError('Rendered trajectory differs from the frozen spatial-window selection')
        selected.extend(ids)
        for row, frame in zip(rows, tr['frames']):
            if (row['timestamp_s'] != frame['timestamp_s']
                    or not np.allclose([row['camera_xy_target'][key] for key in TARGETS], frame['xy'], atol=.001, rtol=0)):
                raise ValueError('Rendered optical-center target differs from frozen schedule')
            path = data / row['mask_path']
            if digest(path) != row['mask_sha256']:
                raise ValueError('Rendered semantic mask differs')
            with Image.open(path) as image:
                visible[row['sample_id']] = bool(np.isin(np.asarray(image), (2, 6)).mean() >= .01)
    if len(selected) != 90:
        raise ValueError('Expected ninety wholly distant full16 windows')
    base_sha = model_manifest_identity(reference, baseline, 'tcn8', reference_sha)
    long_sha = model_manifest_identity(reference, candidate, setting, reference_sha)
    if base_sha != long_sha:
        raise ValueError('The temporal-window comparison must use the same training input')
    comparison = dict(baseline=evaluate(sequences, set(selected), visible, baseline, base_sha, ('cnn1', 'tcn8'), 16),
                      candidate=evaluate(sequences, set(selected), visible, candidate, long_sha, (setting,), 16))
    for group in comparison.values():
        for model in group['models'].values():
            for mode in model['settings'].values():
                stats = mode['metrics']
                stats['all_16_at_least_100mm_from_train'] = stats.pop('all_8_at_least_100mm_from_revised_train')
    record = dict(status='LONG_TEMPORAL_NEURAL_CHALLENGE_COMPLETE', comparison=comparison,
                  test_manifest_sha256=test_sha, plan_sha256=digest(plan_path), test_frames=240,
                  whole16_spatially_unseen=90, whole16_endpoint_ids=selected,
                  source_sha256=digest(Path(__file__)),
                  evaluator_sha256=digest(Path(__file__).with_name('evaluate_geometry.py')),
                  quality='HOLD', native='NOT_RUN')
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8', newline='\n')
    for key, group in comparison.items():
        for name, model in group['models'].items():
            print(key, name, {tag: (m['frames'], round(m['median_mm']), round(m['p95_mm']))
                             for tag, m in model['settings']['chronological']['metrics'].items()
                             if tag in ('all_unseen_trajectories', 'all_complete_windows',
                                        'all_16_at_least_100mm_from_train')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'plan', 'reference', 'field', 'baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setting', choices=('tcn8', 'tcn16'), default='tcn16')
    args = parser.parse_args()
    run(args.data, args.plan, args.reference, args.field, args.baseline, args.candidate, args.out, args.setting)
