"""Validate exact pose pairs and evaluate simulation lighting profiles absent from training."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_challenge import score
from evaluate_neural import digest, load_sequences, model_manifest_identity, read, verified_graph


def run(data, plan_path, reference_data, models, out, setting='tcn8'):
    plan = read(plan_path)
    qa = read(data / 'QA.json')
    sequences, manifest, _ = load_sequences(data)
    original = [json.loads(line) for line in (reference_data / 'manifest.jsonl').read_text().splitlines()]
    training_profiles = {row['appearance'] for row in original if row['split'] == 'train'}
    training_images = {row['image_sha256'] for row in original if row['split'] != 'test'}
    expected = {tr['id'] + '-' + style: (tr, style) for tr in plan['trajectories'] for style in tr['variants']}
    if (qa['plan_sha256'] != digest(plan_path) or manifest != qa['manifest_sha256']
            or {r[0]['trajectory_id'] for r, _ in sequences} != set(expected)
            or qa['frames'] != 200 or len(sequences) != 8
            or {'warm_reserved', 'cool_reserved'} & training_profiles):
        raise ValueError('Expected reserved appearance groups absent from the training manifest')
    paired = {}
    for rows, _ in sequences:
        tr, style = expected[rows[0]['trajectory_id']]
        for row, frame in zip(rows, tr['frames']):
            if (row['image_sha256'] in training_images or row['timestamp_s'] != frame['timestamp_s']
                    or not np.allclose(list(row['camera_xy_target'].values()), frame['xy'], atol=.001, rtol=0)):
                raise ValueError('Reserved light sample disagrees with the frozen pose or duplicates a training image')
            mask = data / row['mask_path']
            if digest(mask) != row['mask_sha256']:
                raise ValueError('Reserved mask differs')
            paired.setdefault((tr['id'], row['frame_index']), []).append(row)
    if len(paired) != 100:
        raise ValueError('Expected a hundred exact pose pairs')
    for a, b in paired.values():
        if a['mask_sha256'] != b['mask_sha256'] or a['image_sha256'] == b['image_sha256']:
            raise ValueError('Appearance pairs must preserve semantics and change RGB')
    collection = read(models / 'MODEL_COLLECTION.json')
    full = digest(reference_data / 'manifest.jsonl')
    trained = model_manifest_identity(reference_data, models, setting, full)
    session, input_name, centers, metrics, _, _ = verified_graph(models, collection, setting, trained)
    results = score(sequences, session, input_name, centers, int(setting.removeprefix('tcn')), metrics['temperature'])
    record = dict(status='STRICT_APPEARANCE_HOLDOUT_COMPLETE', saved_version=collection['source'],
                  results=results, paired_frames=100, seen_training_profiles=sorted(training_profiles),
                  absent_training_profiles=['warm_reserved', 'cool_reserved'],
                  manifest_sha256=manifest, plan_sha256=digest(plan_path), model_onnx_sha256=metrics['onnx_sha256'],
                  source_sha256=digest(Path(__file__)))
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({key: (value['frames'], round(value['p95_mm'])) for key, value in results['slices'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data', 'plan', 'reference', 'models', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--setting', choices=('tcn8', 'tcn16'), default='tcn8')
    args = parser.parse_args()
    run(args.data, args.plan, args.reference, args.models, args.out, args.setting)
