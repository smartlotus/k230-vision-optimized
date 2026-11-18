"""Check an isolated temporal board report and score explicitly synchronized XY."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


TARGETS = ('camera_x_mm', 'camera_y_mm')
SURVEY_REFERENCE = 'DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and -math.inf < value < math.inf


def read_rows(blob):
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def trial_profile(archive, receipt):
    with archive.open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['archive_sha256'],
                'Board trial archive SHA differs from its receipt')
    with zipfile.ZipFile(archive) as package:
        manifest = json.loads(package.read('MANIFEST.json'))
        require(len(package.namelist()) == receipt['members'] == len(manifest) + 1
                and set(package.namelist()) == set(manifest) | {'MANIFEST.json'}, 'Board trial inventory differs')
        for name, identity in manifest.items():
            raw = package.read(name)
            require(len(raw) == identity['bytes'] and sha(raw) == identity['sha256'],
                    f'Board trial member differs: {name}')
        deployment = json.loads(package.read('deployment.json'))
    require(deployment['kmodel']['kmodel_sha256'] == receipt['kmodel_sha256']
            == manifest['camera_xy.kmodel']['sha256'], 'Board trial model identity differs')
    return deployment, manifest['camera_xy.kmodel']['bytes']


def checked_trace(report, trace, deployment, expected_usb, model_bytes):
    profile = deployment['neural_window']
    centers = {tuple(xy) for xy in deployment['identity']['map_centers_mm']}
    require(report['status'] == 'MEASURED' and not report['cleanup_errors'],
            'A complete board run with clean exit is required')
    require(isinstance(report['device'], str) and report['device'].strip(), 'Board firmware identity is missing')
    require(report['kmodel_sha256'] == deployment['kmodel']['kmodel_sha256']
            and report['model_bytes'] == model_bytes, 'Executed KModel differs')
    require(report['window'] == profile['length'] and report['sample_us'] == profile['sample_us'],
            'Board window or sampling cadence differs')
    uvc = report['negotiated_uvc']
    require(uvc['device'] == expected_usb and uvc['size'] == [1280, 720]
            and finite(uvc['fps']) and uvc['fps'] > 0, 'OV2735 USB identity or negotiated mode differs')
    requested, inferred, dropped, skipped, resets = (report[key] for key in
        ('requested_frames', 'frames', 'dropped', 'capture_decimated', 'resets'))
    require(all(type(k) is int and k >= 0 for k in (requested, inferred, dropped, skipped, resets))
            and 1 <= requested <= 3000 and 0 < inferred == len(trace)
            and inferred + dropped + skipped == requested, 'Capture and trace counts do not agree')
    previous_index, previous_time, reset_count = -1, None, 0
    raw_ticks = deque(maxlen=profile['length'])
    for row in trace:
        index, elapsed, ticks = (row[key] for key in ('capture_index', 'captured_since_start_us', 'capture_ticks_us'))
        require(type(index) is int and previous_index < index < requested and type(elapsed) is int and elapsed >= 0
                and type(ticks) is int and ticks >= 0, 'Invalid board capture index or clock')
        if previous_time is not None:
            gap = elapsed - previous_time
            require(gap >= profile['sample_us'], 'Sampled board timestamps violate the configured cadence')
            if gap > profile['max_gap_us']:
                raw_ticks.clear()
                reset_count += 1
        raw_ticks.append(ticks)
        filled = [raw_ticks[0]] * (profile['length'] - len(raw_ticks)) + list(raw_ticks)
        require(row['window_valid_frames'] == len(raw_ticks) and row['window_timestamps_us'] == filled,
                f'Window timestamps or warm-up disagree at capture {index}')
        prediction = row['prediction']
        require(prediction['status'] == 'unlocalized' and all(prediction[key] is None for key in TARGETS),
                'HOLD trial exported an accepted position')
        candidates = prediction['top_k_camera_xy']
        require(finite(prediction['visual_peak_mass']) and 0 < prediction['visual_peak_mass'] <= 1,
                'Invalid candidate peak mass')
        require(len(candidates) == 5 and all(set(point) == set(TARGETS) and tuple(point[key] for key in TARGETS) in centers
                for point in candidates), 'Candidate coordinates differ from the model grid')
        points = [tuple(point[key] for key in TARGETS) for point in candidates]
        require(all(math.dist(point, earlier) >= 100 for i, point in enumerate(points) for earlier in points[:i]),
                'Candidate separation differs from the neural decoder')
        previous_index, previous_time = index, elapsed
    require(reset_count == resets and report['unlocalized'] == inferred
            and report['last_prediction'] == trace[-1]['prediction']
            and report['last_window_valid_frames'] == trace[-1]['window_valid_frames'],
            'Report totals or final prediction differ from the trace')
    latency = report['latency']
    for name in ('capture', 'cycle', 'rgb', 'ai2d', 'neural_window', 'kpu', 'decode'):
        expected = requested if name in ('capture', 'cycle') else inferred
        require(latency[name]['samples'] == expected, f'Latency denominator differs: {name}')
    wall = report['wall_seconds']
    require(finite(wall) and wall > 0 and wall * 1e6 + 1 >= previous_time
            and finite(report['camera_to_result_fps'])
            and math.isclose(report['camera_to_result_fps'], inferred / wall, rel_tol=1e-5),
            'Board throughput summary differs from capture counts and elapsed time')
    return {row['capture_index']: row for row in trace}


def score_survey(trace, survey):
    seen, scores = set(), []
    for row in survey:
        index = row['capture_index']
        require(type(index) is int and index in trace and index not in seen, 'Survey must name a unique inferred capture index')
        seen.add(index)
        captured = trace[index]
        require(row['captured_since_start_us'] == captured['captured_since_start_us'],
                'Survey frame timestamp differs from the board trace')
        target = row['camera_xy_target']
        measurement = row['measurement']
        require(set(target) == set(TARGETS) and all(finite(value) and abs(value) <= 1200 for value in target.values()),
                'Expected camera optical-center XY in FIELD_MAP millimeters')
        require(measurement['reference'] == SURVEY_REFERENCE and all(isinstance(measurement[key], str)
                and measurement[key].strip() for key in ('method', 'record_id')) and finite(measurement['uncertainty_mm'])
                and measurement['uncertainty_mm'] > 0, 'Survey requires its direct measurement method, ID and uncertainty')
        candidate = captured['prediction']['top_k_camera_xy'][0]
        error = math.dist([target[key] for key in TARGETS], [candidate[key] for key in TARGETS])
        scores.append(dict(capture_index=index, candidate_error_mm=error,
                           uncertainty_mm=measurement['uncertainty_mm'], record_id=measurement['record_id']))
    errors = [row['candidate_error_mm'] for row in scores]
    ordered = sorted(errors)
    return dict(labeled_frames=len(errors), unlabeled_frames=len(trace) - len(errors),
                median_mm=statistics.median(errors) if errors else None,
                p95_nearest_rank_mm=ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
                scores=scores)


def run(archive, receipt_path, report_path, trace_path, expected_usb, source, out, survey_path=None):
    require(source in ('k230', 'mock'), 'Specify the board record source')
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(receipt['status'] == 'NEURAL_BOARD_TRIAL_ARCHIVE_VERIFIED', 'Expected a verified trial receipt')
    deployment, model_bytes = trial_profile(archive, receipt)
    report_bytes, trace_bytes = report_path.read_bytes(), trace_path.read_bytes()
    report = json.loads(report_bytes)
    trace = checked_trace(report, read_rows(trace_bytes), deployment, expected_usb, model_bytes)
    survey = survey_path.read_bytes() if survey_path else None
    result = dict(status='BOARD_CAPTURE_RECORDS_CONSISTENT', source=source, qualification='HOLD',
                  trial_receipt_sha256=sha(receipt_bytes), trial_archive_sha256=receipt['archive_sha256'],
                  kmodel_sha256=report['kmodel_sha256'], report_sha256=sha(report_bytes), trace_sha256=sha(trace_bytes),
                  survey_sha256=sha(survey) if survey is not None else None,
                  firmware=report['device'], negotiated_uvc=report['negotiated_uvc'],
                  counters={key: report[key] for key in ('requested_frames', 'frames', 'dropped', 'capture_decimated', 'resets')},
                  wall_seconds=report['wall_seconds'], camera_to_result_fps=report['camera_to_result_fps'],
                  latency=report['latency'], memory={key: report.get(key) for key in ('initial_memory', 'final_memory',
                  'active_memory_samples', 'minimum_python_heap_free', 'active_pool_peaks', 'memory_sampling_failures')},
                  survey=score_survey(trace, read_rows(survey) if survey else []),
                  full_uvc_pixel_equivalence='UNVERIFIED', shared_kpu_scheduler='UNVERIFIED')
    with out.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'receipt', 'report', 'trace', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--usb-id', required=True)
    parser.add_argument('--source', choices=('k230', 'mock'), required=True)
    parser.add_argument('--survey', type=Path)
    args = parser.parse_args()
    result = run(args.archive, args.receipt, args.report, args.trace, args.usb_id, args.source, args.out, args.survey)
    print(json.dumps({key: result[key] for key in ('status', 'source', 'qualification', 'counters', 'survey')}))
