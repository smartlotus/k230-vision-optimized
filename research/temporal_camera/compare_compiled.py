"""Measure the exact quantized model versus its float export on full validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import digest
from localization_runtime import CameraXYTracker
from train_map_temporal import stats


def main():
    parser = argparse.ArgumentParser()
    for name in ('ptq', 'parent', 'manifest', 'float-replay'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.ptq / 'KMODEL_REPORT.json').read_text())
    if report['parent_sha256'] != digest(args.parent) or report['dataset_manifest_sha256'] != digest(args.manifest):
        raise ValueError('Compiler evidence references a different map or dataset')
    scores = np.load(args.ptq / 'SIMULATOR_SCORES.npz', allow_pickle=False)
    rows = [r for line in args.manifest.read_text().splitlines() if (r := json.loads(line))['split'] == 'val']
    ids = [row['sample_id'] for row in rows]
    if ids != scores['sample_ids'].tolist() or ids != [row['sample_id'] for row in report['pairs']]:
        raise ValueError('Expected complete validation in trajectory order')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    centers = parent['centers_mm']
    float_replay = json.loads(args.float_replay.read_text())
    if (float_replay['map_parent_sha256'], float_replay['dataset_manifest_sha256']) != (
        report['parent_sha256'], report['dataset_manifest_sha256']
    ):
        raise ValueError('Float replay references a different map or dataset')

    by_startup = {}
    for name, startup in (('two_zone', 'competition'), ('global', 'global')):
        errors = []
        zone_trials = 0
        zone_correct = 0
        for row, logits in zip(rows, scores['logits']):
            if row['frame_index'] == 0:
                tracker = CameraXYTracker(centers, startup=startup)
            result = tracker.update(logits.tolist(), row['timestamp_s'])
            target = row['camera_xy_target']
            x, y = result['candidate_x_mm'], result['candidate_y_mm']
            errors.append(np.hypot(x - target['camera_x_mm'], y - target['camera_y_mm']))
            if row['frame_index'] == 0:
                zone_trials += 1
                zone_correct += int(y * target['camera_y_mm'] > 0)
        by_startup[name] = dict(simulator=stats(errors), float_replay=float_replay['results'][name],
                                initial_zone_trials=zone_trials, initial_zone_correct_simulator=zone_correct,
                                initial_zone_correct_float=float_replay['initial_zone_correct'][name])
    comparison = {
        'status': 'FULL_VALIDATION_DIAGNOSTIC',
        'evaluation_frames': len(rows),
        'kmodel_sha256': report['kmodel_sha256'],
        'dataset_manifest_sha256': report['dataset_manifest_sha256'],
        'anchor_agreement': report['anchor_agreement'],
        'raw_float_onnx': stats([r['onnx_error_mm'] for r in report['pairs']]),
        'raw_kmodel_simulator': stats([r['kmodel_error_mm'] for r in report['pairs']]),
        'filtered': by_startup,
        'qualification': 'ACCURACY_HOLD',
        'native_k230': 'NOT RUN',
    }
    with (args.ptq / 'VALIDATION_COMPARISON.json').open('x', encoding='utf8') as stream:
        json.dump(comparison, stream, indent=2)
    print(json.dumps(comparison))


if __name__ == '__main__':
    main()
