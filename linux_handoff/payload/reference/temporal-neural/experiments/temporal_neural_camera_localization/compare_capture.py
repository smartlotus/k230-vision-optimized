"""Compare the matched A03 CNN-1 and TCN-8 on the same capture endpoints."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from replay_capture import TARGETS, run, sha


SETTINGS = ('cnn1', 'tcn8')
DATASET_MANIFEST_SHA = '94d12969b1b68122c2ab5193340a05ff05021dc07ace1ab989fa1acea532c22e'
SAVE_IDENTITY = ('/code/lotus30/k230-neural-camera-xy-geometry-a03', 1, 145553018)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def load_model_pair(models, profile_path):
    collection = read(models / 'MODEL_COLLECTION.json')
    source = collection['source']
    if (collection['status'] != 'ORIGINAL_NAMED_MODEL_MEMBERS_SHA256_PASS'
            or (source['ref'], source['versionNumber'], source['kernelId']) != SAVE_IDENTITY):
        raise ValueError('Expected original saved Geometry A03 CNN-1 and TCN-8')
    profile = read(profile_path)
    if profile['window'] != 8 or profile['training_manifest_sha256'] != DATASET_MANIFEST_SHA:
        raise ValueError('Expected the matched A03 eight-frame sampling profile')
    entries = {item['name']: item for item in collection['files']}
    metrics = {}
    for setting in SETTINGS:
        for name in ('camera_xy.onnx', 'METRICS.json'):
            file = models / setting / name
            item = entries[f'models/{setting}/{name}']
            if item['setting'] != setting or file.stat().st_size != item['bytes'] or sha(file) != item['sha256']:
                raise ValueError(f'Original saved {setting} member differs: {name}')
        metrics[setting] = read(models / setting / 'METRICS.json')
    return source, profile, metrics


def paired_endpoints(cnn, tcn):
    fields = ('sample_id', 'sequence_id', 'split', 'start_zone_id', 'timestamp_us', 'image_sha256', 'gap_reset')
    pairs = list(zip(cnn, tcn, strict=True))
    for baseline, temporal in pairs:
        if any(baseline[field] != temporal[field] for field in fields):
            raise ValueError('Models must be scored on exactly the same captured endpoints')
        if ('candidate_error_mm' in baseline) != ('candidate_error_mm' in temporal):
            raise ValueError('Ground-truth coverage differs across models')
        if (baseline['window_timestamps_us'][-1] != temporal['window_timestamps_us'][-1]
                or baseline['window_timestamps_us'][-1] != temporal['timestamp_us']):
            raise ValueError('Model window does not end at the scored capture')
    return pairs


def paired_stats(pairs):
    labeled = [(cnn['candidate_error_mm'], tcn['candidate_error_mm']) for cnn, tcn in pairs
               if 'candidate_error_mm' in cnn]
    values = np.array(labeled, dtype=float).reshape(-1, 2)
    differences = values[:, 1] - values[:, 0]
    result = dict(frames=len(pairs), labeled=len(labeled),
                  sequences=len({tcn['sequence_id'] for _, tcn in pairs}),
                  lower_tcn_error=int((differences < 0).sum()), equal_error=int((differences == 0).sum()),
                  lower_cnn_error=int((differences > 0).sum()),
                  tcn_minus_cnn_median_mm=float(np.median(differences)) if labeled else None)
    for column, setting in enumerate(SETTINGS):
        errors = values[:, column]
        jumps, transitions = 0, 0
        last = None
        for pair in pairs:
            row = pair[column]
            candidate = row['prediction']['top_k_camera_xy'][0]
            xy = tuple(candidate[key] for key in TARGETS)
            preceding_timestamp = pair[1]['window_timestamps_us'][-2]
            if (last is not None and row['sequence_id'] == last[0] and not row['gap_reset']
                    and preceding_timestamp == last[1]):
                transitions += 1
                jumps += math.dist(xy, last[2]) > 500
            last = row['sequence_id'], row['timestamp_us'], xy
        result[setting] = dict(median_mm=float(np.median(errors)) if labeled else None,
                               p95_mm=float(np.quantile(errors, .95)) if labeled else None,
                               jumps_over_500mm=jumps, scored_transitions=transitions)
    return result


def evaluate_test(pairs):
    test = [pair for pair in pairs if pair[1]['split'] == 'test']
    complete = [pair for pair in test if pair[1]['window_valid_frames'] == 8]
    return dict(all_endpoints=paired_stats(test), complete_tcn8_windows=paired_stats(complete),
                by_start_zone={str(zone): dict(
                    all_endpoints=paired_stats([pair for pair in test if pair[1]['start_zone_id'] == zone]),
                    complete_tcn8_windows=paired_stats([pair for pair in complete if pair[1]['start_zone_id'] == zone]))
                    for zone in (1, 2)})


def compare(manifest, models, profile_path, output, capture_source, preprocessing='pil'):
    profile_sha = sha(profile_path)
    source, profile, metrics = load_model_pair(models, profile_path)
    if sha(profile_path) != profile_sha:
        raise ValueError('A03 sampling profile changed while loading')
    output.mkdir(exist_ok=False)
    reports, rows = {}, {}
    for setting in SETTINGS:
        model_profile = dict(profile, window=1 if setting == 'cnn1' else 8,
                             best_sha256=metrics[setting]['best_sha256'],
                             temperature=metrics[setting]['temperature'], threshold=None,
                             threshold_qualification='FIT_FROM_CAPTURE_VALIDATION_ONLY')
        model_profile.pop('validation_threshold_fit', None)
        setting_profile = output / f'{setting}.profile.json'
        setting_profile.write_text(json.dumps(model_profile, indent=2) + '\n', encoding='utf8')
        index = output / f'{setting}.predictions.jsonl'
        reports[setting] = run(manifest, models / setting / 'camera_xy.onnx', setting_profile,
                               models / setting / 'METRICS.json', index, capture_source, preprocessing)
        if reports[setting]['onnx_sha256'] != metrics[setting]['onnx_sha256']:
            raise ValueError('Original model changed during capture replay')
        rows[setting] = [json.loads(line) for line in index.read_text(encoding='utf8').splitlines()]
        if 'splits' not in reports[setting]:
            raise ValueError('Paired evaluation requires a manifest with validation and test sequence IDs')
    if (reports['cnn1']['manifest_sha256'] != reports['tcn8']['manifest_sha256']
            or reports['cnn1']['measured_gt_frames'] != reports['tcn8']['measured_gt_frames']):
        raise ValueError('Capture source changed across the two replays')
    pairs = paired_endpoints(rows['cnn1'], rows['tcn8'])
    result = dict(status='PAIRED_CAPTURE_REPLAY_COMPLETE', source=capture_source, saved_model_source=source,
                  manifest_sha256=reports['tcn8']['manifest_sha256'], input_profile_sha256=profile_sha,
                  comparison_source_sha256=sha(Path(__file__)),
                  replay_source_sha256=sha(Path(__file__).with_name('replay_capture.py')),
                  sample_us=profile['sample_us'], max_gap_us=profile['max_gap_us'],
                  paired_frames=len(pairs), comparison='Raw candidate positions on matched independent test endpoints',
                  host_preprocessing=preprocessing, test=evaluate_test(pairs), per_model=reports,
                  native='NOT_RUN', qualification='HOLD')
    with (output / 'PAIRED_TEST.json').open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'models', 'profile', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', choices=('synthetic', 'ov2735'), required=True)
    parser.add_argument('--preprocessing', choices=('pil', 'ai2d-640'), default='pil')
    args = parser.parse_args()
    result = compare(args.manifest, args.models, args.profile, args.out, args.source, args.preprocessing)
    print(json.dumps({key: result[key] for key in ('status', 'paired_frames', 'test', 'qualification')}))
