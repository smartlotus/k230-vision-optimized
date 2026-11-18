"""Run timestamped RGB manifests through the same fixed-window neural trial."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from capture_pixels import ai2d_640
from neural_decode import decode_logits
from neural_window import FrameWindow


TARGETS = ('camera_x_mm', 'camera_y_mm')
SURVEY_REFERENCE = 'DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY'
TOLERANCE_MM = 200
MIN_VALIDATION_FRAMES = 25
MIN_VALIDATION_PRECISION = .95


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite(value):
    return type(value) in (int, float) and -math.inf < value < math.inf


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def read_capture(manifest, source):
    items = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines() if line.strip()]
    if not items or any(not isinstance(item, dict) for item in items):
        raise ValueError('Capture index must contain JSON objects')
    grouped = any('sequence_id' in item for item in items)
    if source == 'ov2735' and not grouped:
        raise ValueError('OV2735 evaluation requires explicit sequence IDs and split assignments')
    sequences, samples, image_splits, recording_splits = {}, set(), {}, {}
    root = manifest.parent.resolve()
    for item in items:
        sample_id, ts = item.get('sample_id'), item.get('timestamp_s')
        if not nonempty(sample_id) or sample_id in samples or not finite(ts) or not finite(ts * 1e6):
            raise ValueError('Expected unique sample IDs and finite timestamps')
        samples.add(sample_id)
        sequence_id = item.get('sequence_id') if grouped else 'capture'
        split = item.get('split') if grouped else None
        zone = item.get('start_zone_id') if grouped else None
        recording_id = item.get('recording_id')
        if grouped and (not nonempty(sequence_id) or split not in ('val', 'test')
                        or type(zone) is not int or zone not in (1, 2)):
            raise ValueError('Grouped rows require sequence_id, split val/test, and start_zone_id 1/2')
        if source == 'ov2735' and not nonempty(recording_id):
            raise ValueError('OV2735 evaluation requires the original recording_id')
        if recording_id is not None and (not nonempty(recording_id)
                                         or recording_splits.setdefault(recording_id, split) != split):
            raise ValueError('An original recording cannot cross validation and test')
        sequence = sequences.setdefault(sequence_id, dict(split=split, start_zone_id=zone,
                                                          recording_id=recording_id, frames=[]))
        if (sequence['split'], sequence['start_zone_id'], sequence['recording_id']) != (split, zone, recording_id):
            raise ValueError('Split, start zone and recording must stay fixed within a sequence')
        timestamp = int(round(ts * 1e6))
        frames = sequence['frames']
        if frames and timestamp <= frames[-1]['timestamp_us']:
            raise ValueError('Each capture sequence requires strictly increasing microsecond timestamps')
        relative = Path(item['image_path'])
        image = (root / relative).resolve()
        if relative.is_absolute() or not image.is_relative_to(root):
            raise ValueError('Image path must stay inside the capture index directory')
        file_sha = sha(image)
        if item.get('image_sha256', file_sha) != file_sha:
            raise ValueError('Capture RGB differs from the supplied digest')
        if image_splits.setdefault(file_sha, split) != split:
            raise ValueError('Identical RGB bytes cannot cross validation and test')
        truth = item.get('camera_xy_target')
        measurement = item.get('measurement')
        uncertainty = None
        if truth is not None:
            if (not isinstance(truth, dict) or set(truth) != set(TARGETS)
                    or any(not finite(value) or abs(value) > 1200 for value in truth.values())):
                raise ValueError('Expected camera optical-center XY in FIELD_MAP millimeters')
            if source == 'ov2735':
                if (not isinstance(measurement, dict) or measurement.get('reference') != SURVEY_REFERENCE
                        or not nonempty(measurement.get('method')) or not nonempty(measurement.get('record_id'))
                        or not finite(measurement.get('uncertainty_mm')) or measurement['uncertainty_mm'] <= 0):
                    raise ValueError('Each OV2735 truth label requires its direct survey method, record ID and uncertainty')
                uncertainty = measurement['uncertainty_mm']
        elif measurement is not None:
            raise ValueError('Measurement metadata requires its paired camera XY label')
        frames.append(dict(item, timestamp_us=timestamp, image=image, image_sha256=file_sha,
                           uncertainty_mm=uncertainty))
    return sequences, len(items), grouped


def summarize(records, threshold, window):
    labeled = [row for row in records if row.get('candidate_error_mm') is not None]
    errors = [row['candidate_error_mm'] for row in labeled]
    accepted = [row for row in records if threshold is not None and row['prediction']['visual_peak_mass'] >= threshold]
    accepted_errors = [row['candidate_error_mm'] for row in accepted if 'candidate_error_mm' in row]
    return dict(inferred=len(records), labeled=len(errors), complete_windows=sum(
        row['window_valid_frames'] == window for row in records),
        candidate_median_mm=float(np.median(errors)) if errors else None,
        candidate_p95_mm=float(np.quantile(errors, .95)) if errors else None,
        candidate_within_200_count=sum(error <= TOLERANCE_MM for error in errors),
        accepted=len(accepted), accepted_labeled=len(accepted_errors),
        accepted_within_200_count=sum(error <= TOLERANCE_MM for error in accepted_errors))


def fit_threshold(records):
    scores = np.array([row['prediction']['visual_peak_mass'] for row in records if 'candidate_error_mm' in row])
    correct = np.array([row['candidate_error_mm'] <= TOLERANCE_MM for row in records if 'candidate_error_mm' in row])
    for candidate in np.unique(scores):
        accepted = scores >= candidate
        if accepted.sum() >= MIN_VALIDATION_FRAMES and correct[accepted].mean() >= MIN_VALIDATION_PRECISION:
            return float(candidate)
    return None


def blue_zone(candidate):
    x, y = (candidate[key] for key in TARGETS)
    if x >= 900 and abs(y) >= 900:
        return 1 if y > 0 else 2
    return None


def gap_recoveries(records):
    starts = [i for i, row in enumerate(records) if row['gap_reset']]
    recoveries = []
    for start, end in zip(starts, starts[1:] + [len(records)]):
        after_gap = records[start:end]
        stable = 0
        recovered_s = None
        for row in after_gap:
            stable = stable + 1 if row.get('candidate_error_mm', math.inf) <= TOLERANCE_MM else 0
            if stable == 3:
                recovered_s = (row['timestamp_us'] - after_gap[0]['timestamp_us']) / 1e6
                break
        recoveries.append(dict(first_sample_id=after_gap[0]['sample_id'], inferred_after_gap=len(after_gap),
                               labeled_after_gap=sum('candidate_error_mm' in row for row in after_gap),
                               candidate_recovery_s=recovered_s))
    return recoveries


def grouped_report(predictions, summaries, all_records, window):
    validation = [row for row in all_records if row['split'] == 'val']
    threshold = fit_threshold(validation)
    for row in all_records:
        row['accepted_by_validation_threshold'] = (threshold is not None
                                                  and row['prediction']['visual_peak_mass'] >= threshold)
    splits = {}
    for split in ('val', 'test'):
        by_zone = {}
        for zone in (1, 2):
            runs = [records for key, records in predictions.items()
                    if summaries[key]['split'] == split and summaries[key]['start_zone_id'] == zone]
            stats = summarize([row for records in runs for row in records], threshold, window)
            by_zone[str(zone)] = dict(stats, sequences=len(runs), correct_start_zone_candidates=sum(
                blue_zone(records[0]['prediction']['top_k_camera_xy'][0]) == zone for records in runs))
        split_rows = [row for row in all_records if row['split'] == split]
        splits[split] = dict(summarize(split_rows, threshold, window), by_start_zone=by_zone)
    for key, records in predictions.items():
        summaries[key].update(summarize(records, threshold, window),
                              candidate_start_zone=blue_zone(records[0]['prediction']['top_k_camera_xy'][0]),
                              gap_recoveries=gap_recoveries(records))
    return dict(validation_threshold_fit=dict(threshold=threshold, source_split='val', tolerance_mm=TOLERANCE_MM,
                                               minimum_labeled_accepted=MIN_VALIDATION_FRAMES,
                                               minimum_precision=MIN_VALIDATION_PRECISION),
                splits=splits, sequences=summaries)


def run(manifest, onnx_path, profile_path, metrics_path, out, source='unspecified', preprocessing='pil'):
    if source not in ('unspecified', 'synthetic', 'ov2735'):
        raise ValueError('Invalid capture source')
    if preprocessing not in ('pil', 'ai2d-640'):
        raise ValueError('Invalid capture preprocessing')
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    metrics = json.loads(metrics_path.read_text(encoding='utf8'))
    model_bytes = onnx_path.read_bytes()
    onnx_sha = hashlib.sha256(model_bytes).hexdigest()
    if (onnx_sha != metrics['onnx_sha256'] or metrics['best_sha256'] != profile['best_sha256']
            or metrics['temperature'] != profile['temperature']):
        raise ValueError('Neural graph, selected model and runtime profile differ')
    receipt = out.with_suffix(out.suffix + '.report.json')
    if out.exists() or receipt.exists():
        raise FileExistsError('Choose a fresh predictions and report path')
    manifest_sha = sha(manifest)
    sequences, total, grouped = read_capture(manifest, source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=['CPUExecutionProvider'])
    input_, = session.get_inputs()
    output, = session.get_outputs()
    shape = [1, profile['window'] * 3, 96, 160]
    if input_.type != 'tensor(float)' or input_.shape != shape or output.shape != [1, 2304]:
        raise ValueError('Capture neural graph input/output differs')
    axis = range(-1175, 1200, 50)
    centers = [[x, y] for y in axis for x in axis]
    predictions = {}
    summaries = {}
    for sequence_id, sequence in sequences.items():
        ring = FrameWindow(np, profile['window'], profile['sample_us'], profile['max_gap_us'], lambda a, b: a - b)
        records = predictions[sequence_id] = []
        for item in sequence['frames']:
            timestamp = item['timestamp_us']
            if not ring.due(timestamp):
                continue
            image_bytes = item['image'].read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != item['image_sha256']:
                raise ValueError('Capture RGB changed during replay')
            with Image.open(io.BytesIO(image_bytes)) as im:
                if source == 'ov2735' and im.width * 9 != im.height * 16:
                    raise ValueError('The current OV2735 preprocessing expects 16:9 RGB')
                if preprocessing == 'ai2d-640':
                    if im.format != 'PNG' or im.mode != 'RGB' or im.size != (640, 360):
                        raise ValueError('AI2D replay requires a lossless 640x360 RGB PNG intermediate')
                    pixel = ai2d_640(np.asarray(im).transpose(2, 0, 1))[None]
                else:
                    scaled = im.convert('RGB').resize((160, 90), Image.Resampling.BILINEAR)
                    pixel = np.asarray(ImageOps.expand(scaled, (0, 3, 0, 3), fill=(114, 114, 114)))
                    pixel = pixel.transpose(2, 0, 1)[None].copy()
                previous_resets = ring.resets
                packed = ring.append(pixel, timestamp)
            logits, = session.run(None, {input_.name: packed.astype(np.float32) / 255})
            decoded = decode_logits(logits[0].tolist(), centers, profile['temperature'], profile['threshold'])
            record = dict(sample_id=item['sample_id'], timestamp_s=item['timestamp_s'], image_sha256=item['image_sha256'],
                          window_timestamps_us=list(ring.tensor_timestamps_us), window_valid_frames=ring.count,
                          prediction=decoded)
            if grouped:
                record.update(sequence_id=sequence_id, split=sequence['split'], start_zone_id=sequence['start_zone_id'],
                              timestamp_us=timestamp, gap_reset=ring.resets > previous_resets)
            if item.get('camera_xy_target') is not None:
                truth = item['camera_xy_target']
                candidate = decoded['top_k_camera_xy'][0]
                record['candidate_error_mm'] = float(np.hypot(candidate['camera_x_mm'] - truth['camera_x_mm'],
                                                              candidate['camera_y_mm'] - truth['camera_y_mm']))
                if item['uncertainty_mm'] is not None:
                    record['survey_uncertainty_mm'] = item['uncertainty_mm']
            records.append(record)
        summaries[sequence_id] = dict(split=sequence['split'], start_zone_id=sequence['start_zone_id'],
                                      recording_id=sequence['recording_id'],
                                      frames_in_index=len(sequence['frames']), window_resets=ring.resets,
                                      decimated=len(sequence['frames']) - len(records))
    all_records = [row for records in predictions.values() for row in records]
    aggregate = summarize(all_records, profile['threshold'], profile['window'])
    report = dict(status='TIMESTAMPED_NEURAL_RGB_REPLAY_COMPLETE', capture_source=source, frames_in_index=total,
                  inferred=len(all_records), decimated=total - len(all_records),
                  window_resets=sum(row['window_resets'] for row in summaries.values()),
                  onnx_sha256=onnx_sha, profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
                  manifest_sha256=manifest_sha,
                  provided_gt_frames=aggregate['labeled'],
                  measured_gt_frames=sum('survey_uncertainty_mm' in row for row in all_records),
                  candidate_median_mm=aggregate['candidate_median_mm'], candidate_p95_mm=aggregate['candidate_p95_mm'],
                  native_preprocessing='NOT_RUN', qualification='HOLD')
    if preprocessing == 'ai2d-640':
        report['host_preprocessing'] = 'AI2D_640X360_HALF_PIXEL_MIRROR'
    if grouped:
        report['aggregate_scope'] = 'pooled_validation_and_test'
        report.update(grouped_report(predictions, summaries, all_records, profile['window']))
    if sha(manifest) != manifest_sha:
        raise ValueError('Capture index changed during replay')
    partial = out.with_suffix(out.suffix + '.partial')
    pending_report = receipt.with_suffix(receipt.suffix + '.partial')
    if partial.exists() or pending_report.exists():
        raise FileExistsError('Capture replay already has temporary output at this path')
    wrote_predictions = wrote_report = published_predictions = False
    try:
        with partial.open('x', encoding='utf8', newline='\n') as stream:
            wrote_predictions = True
            if out.exists() or receipt.exists():
                raise FileExistsError('Choose a fresh predictions and report path')
            for row in all_records:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        report['predictions_sha256'] = sha(partial)
        with pending_report.open('x', encoding='utf8', newline='\n') as stream:
            wrote_report = True
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        partial.rename(out)
        published_predictions = True
        pending_report.rename(receipt)
    except BaseException:
        if published_predictions:
            out.unlink()
        raise
    finally:
        if wrote_predictions and not published_predictions:
            partial.unlink(missing_ok=True)
        if wrote_report:
            pending_report.unlink(missing_ok=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'onnx', 'profile', 'metrics', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', choices=('unspecified', 'synthetic', 'ov2735'), default='unspecified')
    parser.add_argument('--preprocessing', choices=('pil', 'ai2d-640'), default='pil')
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.onnx, args.profile, args.metrics, args.out,
                         args.source, args.preprocessing)))
