"""Replay timestamped camera RGB using the released CameraXY models on a host."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image

from board_multiroute import RGBWindow
from camera_preprocess import ai2d_downsample_4x, rgb_uint8
from competition_paths import ZONE_CENTERS, ZONE_HALF_WIDTH
from dataset import digest
from localization_runtime import CameraXYTracker


WINDOWS = {'cnn1': 1, 'gru4': 4, 'gru8': 8, 'tcn4': 4, 'tcn8': 8}
BASE_FIELDS = {'trajectory_id', 'frame_index', 'timestamp_s', 'image_path'}
EXTRA_FIELDS = {'image_sha256', 'camera_xy_target', 'measurement', 'start_zone', 'first_landmark'}
TARGETS = ('camera_x_mm', 'camera_y_mm')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_survey(row):
    target = row.get('camera_xy_target')
    measurement = row.get('measurement')
    if target is None and measurement is None:
        return
    if not isinstance(target, dict) or set(target) != set(TARGETS):
        raise ValueError('Only camera_x_mm and camera_y_mm are allowed in a survey position')
    if any(not finite(v) or not -1200 <= v < 1200 for v in target.values()):
        raise ValueError('Survey XY must be finite FIELD_MAP camera optical-center coordinates')
    if not isinstance(measurement, dict) or set(measurement) != {'reference', 'uncertainty_mm', 'method', 'record_id'}:
        raise ValueError('Supply the matching optical-center measurement')
    if (measurement['reference'] != 'DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY'
            or not finite(measurement['uncertainty_mm']) or measurement['uncertainty_mm'] <= 0
            or any(not isinstance(measurement[k], str) or not measurement[k].strip() for k in ('method', 'record_id'))):
        raise ValueError('Invalid survey provenance or uncertainty')


def load_trajectories(manifest):
    groups = {}
    seen_paths = set()
    root = manifest.parent.resolve()
    with manifest.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or not BASE_FIELDS <= set(row) <= BASE_FIELDS | EXTRA_FIELDS:
                raise ValueError('Unexpected or missing replay manifest fields')
            identifier = row['trajectory_id']
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError('Each clip needs a nonempty trajectory_id')
            frames = groups.setdefault(identifier, [])
            index, stamp = row['frame_index'], row['timestamp_s']
            if type(index) is not int or index != len(frames):
                raise ValueError(f'Frame indexes must start at 0 and be contiguous in {identifier}')
            if not finite(stamp) or stamp < 0 or (frames and stamp <= frames[-1]['timestamp_s']):
                raise ValueError(f'Capture timestamps must strictly increase in {identifier}')
            if 'start_zone' in row and (index != 0 or type(row['start_zone']) is not int or row['start_zone'] not in (1, 2)):
                raise ValueError('start_zone must be 1 or 2 on frame 0')
            if 'first_landmark' in row and (row['first_landmark'] is not True or any(r.get('first_landmark') for r in frames)):
                raise ValueError('first_landmark may mark exactly one frame per trajectory')
            validate_survey(row)
            image = row['image_path']
            if image is None:
                if any(k in row for k in ('image_sha256', 'camera_xy_target', 'first_landmark')):
                    raise ValueError('A missing capture has no image, landmark or surveyed frame')
            else:
                if not isinstance(image, str) or not image.strip() or Path(image).is_absolute() or '..' in Path(image).parts:
                    raise ValueError('Enter a relative image path under the manifest directory')
                path = (root / image).resolve()
                if not path.is_relative_to(root) or not path.is_file() or path in seen_paths:
                    raise ValueError('Missing, reused or out-of-root capture path')
                seen_paths.add(path)
                row['resolved_path'] = path
                sha = digest(path)
                if row.get('image_sha256', sha) != sha:
                    raise ValueError(f'Capture hash differs for {image}')
                row['image_sha256'] = sha
            frames.append(row)
    if not groups or any(not any(r['image_path'] for r in frames) for frames in groups.values()):
        raise ValueError('Each trajectory needs at least one observed RGB frame')
    return groups


def make_model_input(path):
    with Image.open(path) as image:
        width, height = image.size
        if width * 9 != height * 16:
            raise ValueError(f'Capture {path.name} is not a 16:9 input')
        if (width, height) == (160, 90):
            return rgb_uint8(path), 'UNPADDED_RGB160X90'
        rgb = image.convert('RGB')
        if rgb.size == (640, 360):
            policy = 'RGB640X360_AI2D_EQUIVALENT'
        else:
            rgb = rgb.resize((640, 360), Image.Resampling.BILINEAR)
            policy = 'HOST_RESIZE_TO_640X360_THEN_AI2D_EQUIVALENT'
        return ai2d_downsample_4x(np.asarray(rgb).transpose(2, 0, 1)), policy


def checked_member(archive, inventory, name):
    entry = inventory[name]
    info = archive.getinfo(name)
    if info.file_size != entry['bytes'] or info.file_size > 12 * 1024**2:
        raise ValueError(f'Archive size differs for {name}')
    raw = archive.read(name)
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f'Archive member hash differs for {name}')
    return raw


def load_routes(model_zip, board_zip, settings):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    with zipfile.ZipFile(model_zip) as models, zipfile.ZipFile(board_zip) as board:
        mm = json.loads(models.read('SHA256_MANIFEST.json'))
        bm = json.loads(board.read('SHA256_MANIFEST.json'))
        collection = json.loads(checked_member(models, mm, 'models/MODEL_COLLECTION.json'))
        source = collection['source']
        if (source['kernelId'], source['versionNumber']) != (145538534, 1):
            raise ValueError('Use the a06 trained model delivery')
        provenance = {r['name']: r['sha256'] for r in collection['files']}
        base = 'k230_temporal_v1_gray_cnn1/'
        deployment = json.loads(checked_member(board, bm, base + 'deployment.json'))
        compiled = json.loads(checked_member(board, bm, base + 'KMODEL_REPORT.json'))
        anchors = deployment['identity']['map_centers_mm']
        config = deployment['temporal_filter']
        if (compiled['float_onnx_sha256'] != provenance['models/cnn1/camera_xy.onnx']
                or deployment['kmodel']['kmodel_sha256'] != compiled['kmodel_sha256']
                or config.get('acceptance') is not None or config['grid_size'] != 48):
            raise ValueError('Board archive differs from the experimental a06 model contract')
        routes = {}
        for setting in settings:
            name = f'models/{setting}/camera_xy.onnx'
            metrics = f'models/{setting}/METRICS.json'
            if any(mm[p]['sha256'] != provenance[p] for p in (name, metrics)):
                raise ValueError(f'Saved model member differs from Kaggle receipt for {setting}')
            info = json.loads(checked_member(models, mm, metrics))
            net = ort.InferenceSession(checked_member(models, mm, name), sess_options=options,
                                       providers=['CPUExecutionProvider'])
            shape = [1, 3 * WINDOWS[setting], 96, 160]
            if len(net.get_inputs()) != 1 or net.get_inputs()[0].shape != shape:
                raise ValueError(f'Unexpected ONNX input for {setting}')
            routes[setting] = {'net': net, 'config': {**config, 'temperature': info['temperature']},
                               'sha256': provenance[name]}
    return routes, anchors, source


def blue_zone_cells(tracker):
    ids = [[], []]
    for i in range(tracker.grid_size**2):
        row, col = divmod(i, tracker.grid_size)
        x = -1200 + (col + .5) * tracker.cell
        y = -1200 + (row + .5) * tracker.cell
        for zone, (cx, cy) in enumerate(ZONE_CENTERS):
            if max(abs(x - cx), abs(y - cy)) <= ZONE_HALF_WIDTH + 100:
                ids[zone].append(i)
    return ids


def summarize(results):
    def errors(values):
        if not values:
            return {'n': 0, 'status': 'NOT RUN'}
        return {'n': len(values), 'median_mm': float(np.median(values)), 'p95_mm': float(np.quantile(values, .95))}
    observed = [r for r in results if r['image_sha256'] is not None]
    claimed = [r for r in observed if r['survey_error_mm'] is not None]
    checked_start = [r for r in results if r['start_zone_reference'] is not None]
    return {'capture_attempts': len(results), 'observed_frames': len(observed),
            'missing_captures': len(results) - len(observed),
            'lost_updates': sum(r['state'] == 'LOST' for r in results),
            'global_recovery_events': sum(r['relocalized'] for r in results),
            'history_resets': sum(r['history_reset'] for r in results),
            'start_zone_trials': len(checked_start),
            'correct_start_zone': sum(r['start_zone_prediction'] == r['start_zone_reference'] for r in checked_start),
            'undecided_start_zone': sum(r['start_zone_prediction'] is None for r in checked_start),
            'surveyed_raw': errors([r['raw_survey_error_mm'] for r in claimed]),
            'surveyed_filtered': errors([r['survey_error_mm'] for r in claimed]),
            'surveyed_after_landmark': errors([r['survey_error_mm'] for r in claimed if r['at_or_after_landmark']])}


def replay(manifest, output, routes, anchors, source, declared_source):
    if output.exists():
        raise FileExistsError(output)
    groups = load_trajectories(manifest)
    report = {'status': 'HOST_REPLAY_COMPLETE', 'declared_source': declared_source,
              'manifest_sha256': digest(manifest), 'model_source': source, 'native_k230': 'NOT RUN',
              'confidence': 'UNCALIBRATED_POSTERIOR_MASS', 'routes': {}}
    rows_by_route = {name: {} for name in routes}
    for identifier, frames in groups.items():
        trackers = {name: CameraXYTracker(anchors, **item['config']) for name, item in routes.items()}
        zone_indices = {name: blue_zone_cells(tracker) for name, tracker in trackers.items()}
        histories = {name: RGBWindow(np.empty((3 * WINDOWS[name], 96, 160), np.uint8), WINDOWS[name])
                     for name in routes}
        streams = {name: [] for name in routes}
        missing = 0
        landmark = False
        declared_zone = frames[0].get('start_zone')
        first_observed = True
        for row in frames:
            image, policy = make_model_input(row['resolved_path']) if row['image_path'] else (None, None)
            missing = missing + 1 if image is None else 0
            landmark = landmark or row.get('first_landmark', False)
            for name, item in routes.items():
                history = histories[name]
                history_reset = missing == 5 and history.length > 1
                if history_reset:
                    history.clear()
                logits = None
                if image is not None:
                    history.push(image)
                    x = history.packed[None].astype(np.float32) / 255.
                    y = item['net'].run(None, {'rgb_window': x})[0]
                    if y.shape != (1, len(anchors)) or not np.isfinite(y).all():
                        raise ValueError(f'{name} returned invalid XY anchor scores')
                    logits = y[0].tolist()
                tracker = trackers[name]
                state = tracker.update(logits, row['timestamp_s'], visual_quality=0. if image is None else 1.)
                mass = [sum(tracker.filter.belief[i] for i in ids) for ids in zone_indices[name]]
                candidate = [state[k] for k in ('candidate_x_mm', 'candidate_y_mm')] if state['state'] != 'LOST' else None
                raw = list(anchors[int(np.argmax(logits))]) if logits is not None else None
                gt = row.get('camera_xy_target')
                truth = [gt[k] for k in TARGETS] if gt else None
                zone = (1 if mass[0] > mass[1] else 2) if abs(mass[0] - mass[1]) > 1e-9 else None
                streams[name].append({
                    'frame_index': row['frame_index'], 'timestamp_s': row['timestamp_s'],
                    'image_path': row['image_path'], 'image_sha256': row.get('image_sha256'),
                    'preprocessing': policy, 'raw_camera_xy_candidate_mm': raw, 'filtered_camera_xy_candidate_mm': candidate,
                    'position_valid': state['position_valid'], 'state': state['state'], 'relocalized': state['relocalized'],
                    'history_reset': history_reset, 'blue_region_posterior': mass,
                    'posterior_local_mass': state['confidence_local_mass'], 'posterior_far_mass': state['confidence_far_mass'],
                    'start_zone_reference': declared_zone if first_observed and image is not None else None,
                    'start_zone_prediction': zone if first_observed and image is not None else None,
                    'at_or_after_landmark': landmark,
                    'survey_error_mm': math.dist(candidate, truth) if truth and candidate else None,
                    'raw_survey_error_mm': math.dist(raw, truth) if truth and raw else None,
                    'measurement': row.get('measurement'),
                })
            if image is not None:
                first_observed = False
        for name, stream in streams.items():
            rows_by_route[name][identifier] = {'summary': summarize(stream), 'frames': stream}
    for name, trajectories in rows_by_route.items():
        combined = [frame for tr in trajectories.values() for frame in tr['frames']]
        report['routes'][name] = {'onnx_sha256': routes[name]['sha256'], 'summary': summarize(combined),
                                  'trajectories': trajectories}
    handle, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        if os.name == 'nt':
            os.rename(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'models', 'board', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--settings', nargs='+', choices=WINDOWS, default=['cnn1', 'tcn4'])
    parser.add_argument('--source', choices=('ov2735-recording', 'simulated'), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    routes, anchors, source = load_routes(args.models, args.board, args.settings)
    report = replay(args.manifest, args.out, routes, anchors, source, args.source)
    print(json.dumps({name: route['summary'] for name, route in report['routes'].items()}))


if __name__ == '__main__':
    main()
