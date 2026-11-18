"""Stress the temporal neural outputs and qualify selective XY on validation only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_neural import TARGETS, clip, digest, load_sequences, model_manifest_identity, read, score_grid, verified_graph


def replay(sequences, session, input_name, centers, temperature, window, mode):
    records = []
    for rows, frames in sequences:
        first = rows[0]['timestamp_s']
        observations, timestamps = [], []
        for i, (row, frame) in enumerate(zip(rows, frames)):
            if mode == 'drop' and 6 <= i <= 11:
                continue
            if mode == 'blackout' and 6 <= i <= 11:
                frame = np.zeros_like(frame)
            observations.append(frame)
            timestamps.append(row['timestamp_s'])
            packed = clip(observations, len(observations) - 1, window, 'chronological')
            if len(observations) > window:
                del observations[0]
                del timestamps[0]
            logits, = session.run(None, {input_name: packed})
            target = np.array([row['camera_xy_target'][key] for key in TARGETS])
            _, error, score = score_grid(logits[0], centers, target, temperature, 48)
            records.append(dict(trajectory=row['trajectory_id'], frame=i, error_mm=error, score=score,
                                elapsed_s=row['timestamp_s'] - first, span_s=timestamps[-1] - timestamps[0]))
    return records


def calibrate(records):
    # Fit coverage-constrained selective acceptance on validation trajectories alone.
    scores = np.array([r['score'] for r in records])
    good = np.array([r['error_mm'] <= 200 for r in records])
    best = None
    for threshold in np.unique(scores):
        selected = scores >= threshold
        if selected.sum() >= 25 and good[selected].mean() >= .95:
            best = float(threshold)
            break
    return best


def metrics(records, threshold):
    errors = np.array([r['error_mm'] for r in records])
    spans = np.array([r['span_s'] for r in records if r['frame'] >= 7])
    result = dict(frames=len(errors), median_mm=float(np.median(errors)), p95_mm=float(np.quantile(errors, .95)),
                  far_error_rate=float((errors > 500).mean()), median_window_span_s=float(np.median(spans)),
                  accepted_frames=None, accepted_far_errors=None, accepted_precision_within200=None)
    if threshold is not None:
        accepted = np.array([r['score'] >= threshold for r in records])
        result.update(accepted_frames=int(accepted.sum()), accepted_far_errors=int((errors[accepted] > 500).sum()),
                      accepted_precision_within200=float((errors[accepted] <= 200).mean()) if accepted.any() else None)
    return result


def recovery(records):
    delays = []
    groups = {}
    for row in records:
        if row['frame'] >= 12:
            groups.setdefault(row['trajectory'], []).append(row)
    for sequence in groups.values():
        for triple in zip(sequence, sequence[1:], sequence[2:]):
            if all(row['error_mm'] <= 200 for row in triple):
                delays.append(triple[-1]['elapsed_s'] - sequence[0]['elapsed_s'])
                break
    return dict(trajectories=len(groups), recovered_trajectories=len(delays),
                median_seconds=float(np.median(delays)) if delays else None)


def run(data, models, out):
    torch.set_num_threads(2)
    val, dataset_sha, field_sha = load_sequences(data, 'val')
    test, test_sha, _ = load_sequences(data)
    if dataset_sha != test_sha:
        raise ValueError('Validation and test must use the same manifest identity')
    collection = read(models / 'MODEL_COLLECTION.json')
    results = {}
    for setting, window in (('cnn1', 1), ('tcn8', 8)):
        trained = model_manifest_identity(data, models, setting, dataset_sha)
        session, name, centers, meta, _, _ = verified_graph(models, collection, setting, trained)
        cfg = (session, name, centers, meta['temperature'], window)
        val_records = replay(val, *cfg, 'clean')
        threshold = calibrate(val_records)
        results[setting] = dict(validation_threshold=threshold, validation=metrics(val_records, threshold), test={})
        for mode in ('clean', 'drop', 'blackout'):
            rows = replay(test, *cfg, mode)
            recovery_slice = [row for row in rows if row['frame'] >= 12]
            results[setting]['test'][mode] = dict(all_available=metrics(rows, threshold),
                                                 same_post_fault_endpoints=metrics(recovery_slice, threshold),
                                                 recovery=recovery(rows))
    value = dict(status='NEURAL_FAULT_AND_CALIBRATION_AUDIT_COMPLETE', source=collection['source'],
                 dataset_manifest_sha256=dataset_sha, field_mask_sha256=field_sha,
                 source_sha256=digest(Path(__file__)), window_spans_based_on_manifest_timestamps=True,
                 fit=dict(precision=.95, minimum_accepted=25, tolerance_mm=200, source_split='val'),
                 faults=dict(frames=[6, 7, 8, 9, 10, 11], recovery=dict(start_frame=12, frames_in_tolerance=3,
                                                                       tolerance_mm=200)),
                 results=results, native_k230='NOT_RUN')
    out.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({k: v['test']['drop']['recovery'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.models, args.out)
