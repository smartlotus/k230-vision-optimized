"""Replay the deployed frame ring, sampling cadence, reset and neural decoder."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_neural import (TARGETS, digest, load_sequences, model_manifest_identity,
                            read, statistics, verified_graph, SETTINGS)
from neural_decode import decode_logits
from neural_window import FrameWindow


def cadence(data, models, setting, out):
    training, full_sha, _ = load_sequences(data, 'train')
    trained_sha = model_manifest_identity(data, models, setting, full_sha)
    collection = read(models / 'MODEL_COLLECTION.json')
    _, _, _, metrics, _, _ = verified_graph(models, collection, setting, trained_sha)
    intervals = np.concatenate([np.diff([r['timestamp_s'] for r in rows]) for rows, _ in training])
    if not np.isfinite(intervals).all() or intervals.min() <= 0:
        raise ValueError('Expected strictly positive synthetic acquisition intervals')
    sample_us = max(1, int(np.floor(intervals.min() * 1000000)) - 1)
    profile = dict(window=SETTINGS[setting][1], sample_us=sample_us, max_gap_us=max(700000, int(intervals.max() * 1.15e6)),
                   training_manifest_sha256=trained_sha, full_manifest_sha256=full_sha,
                   best_sha256=metrics['best_sha256'], temperature=metrics['temperature'],
                   threshold=None, threshold_qualification='HOLD_VALIDATION_REQUIRES_95PCT_WITHIN200',
                   simulation_only=True, measured_acquisition_cadence='UNKNOWN',
                   train_intervals_s=dict(min=float(intervals.min()), median=float(np.median(intervals)),
                                          max=float(intervals.max())))
    out.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf8', newline='\n')
    return profile


def replay(sequences, model, input_name, centers, profile, mode, fit_validation=False):
    metrics, resets, skipped, span = [], 0, 0, []
    recovery, count = [], 0
    for trajectory_index, (rows, images) in enumerate(sequences):
        buffer = FrameWindow(np, profile['window'], profile['sample_us'], profile['max_gap_us'],
                             lambda current, before: current - before)
        previous, observed_times, series = None, [], []
        for i, row in enumerate(rows):
            timestamp = int(row['timestamp_s'] * 1000000)
            if mode == 'variable_dt':
                timestamp = timestamp + (i // 4) * 60000
            if mode == 'drop' and 6 <= i <= 11:
                continue
            frame = np.rint(images[i] * 255).astype(np.uint8)[None]
            truth = row
            if mode == 'blackout' and 6 <= i <= 11:
                frame.fill(0)
            if mode == 'teleport' and i >= 12:
                alternative = sequences[(trajectory_index + 1) % len(sequences)]
                truth = alternative[0][i]
                frame = np.rint(alternative[1][i] * 255).astype(np.uint8)[None]
            if not buffer.due(timestamp):
                skipped += 1
                continue
            before_resets = buffer.resets
            packed = buffer.append(frame, timestamp)
            if buffer.resets != before_resets:
                observed_times = []
            observed_times.append(timestamp)
            observed_times = observed_times[-profile['window']:]
            logits, = model.run(None, {input_name: packed.astype(np.float32) / 255})
            answer = decode_logits(logits[0].tolist(), centers.tolist(), profile['temperature'], profile['threshold'])
            predicted = np.asarray([answer['top_k_camera_xy'][0][key] for key in TARGETS])
            target = np.asarray([truth['camera_xy_target'][key] for key in TARGETS])
            error = float(np.linalg.norm(predicted - target))
            jump = float(np.linalg.norm(predicted - previous)) if previous is not None else 0.
            previous = predicted
            metrics.append(dict(error_mm=error, confidence=answer['visual_peak_mass'], jump_mm=jump))
            series.append(dict(index=i, timestamp_us=timestamp, error=error))
            count += answer['status'] == 'localized'
            if buffer.count == profile['window']:
                span.append((observed_times[-1] - observed_times[0]) / 1e6)
        resets += buffer.resets
        resumed = [r for r in series if r['index'] >= 12]
        for triple in zip(resumed, resumed[1:], resumed[2:]):
            if all(r['error'] <= 200 for r in triple):
                recovery.append((triple[-1]['timestamp_us'] - resumed[0]['timestamp_us']) / 1e6)
                break
    result = dict(metrics=statistics(metrics), window_resets=resets, decimated_frames=skipped,
                accepted_frames=count, complete_windows=len(span), complete_window_span_median_s=float(np.median(span)) if span else None,
                recovered_trajectories=len(recovery), trajectories=len(sequences),
                recovered_median_s=float(np.median(recovery)) if recovery else None)
    if fit_validation:
        scores = np.array([row['confidence'] for row in metrics])
        correct = np.array([row['error_mm'] <= 200 for row in metrics])
        threshold = None
        for choice in np.unique(scores):
            accepted = scores >= choice
            if accepted.sum() >= 25 and correct[accepted].mean() >= .95:
                threshold = float(choice)
                break
        result['threshold_fit'] = dict(threshold=threshold, source='val', tolerance_mm=200,
                                       minimum_samples=25, minimum_precision=.95)
    return result


def run(data, models, profile_path, out, setting='tcn8'):
    torch.set_num_threads(2)
    profile = cadence(data, models, setting, profile_path)
    sequences, full_sha, _ = load_sequences(data)
    collection = read(models / 'MODEL_COLLECTION.json')
    session, input_name, centers, _, _, _ = verified_graph(models, collection, setting, profile['training_manifest_sha256'])
    validation, _, _ = load_sequences(data, 'val')
    validation_report = replay(validation, session, input_name, centers, profile, 'clean', fit_validation=True)
    profile['validation_threshold_fit'] = validation_report['threshold_fit']
    profile_path.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf8', newline='\n')
    results = {mode: replay(sequences, session, input_name, centers, profile, mode)
               for mode in ('clean', 'drop', 'blackout', 'variable_dt', 'teleport')}
    schedules = {}
    for mode in ('clean', 'drop'):
        sampled = skipped = resets = 0
        for rows, _ in sequences:
            last = None
            for i, row in enumerate(rows):
                if mode == 'drop' and 6 <= i <= 11:
                    continue
                timestamp = int(row['timestamp_s'] * 1000000)
                if last is not None and timestamp - last < profile['sample_us']:
                    skipped += 1
                    continue
                resets += last is not None and timestamp - last > profile['max_gap_us']
                sampled += 1
                last = timestamp
        schedules[mode] = dict(sampled_frames=sampled, decimated_frames=skipped, window_resets=resets)
        if (results[mode]['decimated_frames'] != skipped or results[mode]['window_resets'] != resets
                or results[mode]['metrics']['frames'] != sampled):
            raise ValueError('Runtime acquisition count or reset differs from the timestamp schedule')
    record = dict(status='NEURAL_RUNTIME_RING_REPLAYED', results=results,
                  validation=validation_report,
                  expected_drop_window_resets=schedules['drop']['window_resets'], timestamp_schedules=schedules,
                  model_source=collection['source'], full_test_manifest_sha256=full_sha,
                  profile_sha256=digest(profile_path),
                  source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                 ('audit_runtime.py', 'neural_window.py', 'neural_decode.py')}, native='NOT_RUN')
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({mode: dict(p95=round(result['metrics']['p95_mm']),
                                 resets=result['window_resets'], recovered=result['recovered_trajectories'])
                      for mode, result in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'profile', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setting', choices=SETTINGS, default='tcn8')
    args = parser.parse_args()
    run(args.data, args.models, args.profile, args.out, args.setting)
