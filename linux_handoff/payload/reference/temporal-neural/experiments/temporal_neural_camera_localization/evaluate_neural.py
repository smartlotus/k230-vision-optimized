"""Replay the saved neural graphs on identical causal, camera-XY-only endpoints."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'camera_xy_localization_v2'))
from model import pixels


SETTINGS = {'cnn1': ('cnn', 1), 'gru4': ('gru', 4), 'gru8': ('gru', 8),
            'tcn4': ('tcn', 4), 'tcn8': ('tcn', 8), 'tcn16': ('tcn', 16)}
TARGETS = ('camera_x_mm', 'camera_y_mm')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def model_manifest_identity(data, models, setting, full_manifest_sha):
    """Bind a model to the full pilot or its exact cloud train/validation subset."""
    checkpoint = torch.load(models / setting / 'best.pt', weights_only=True, map_location='cpu')
    trained = checkpoint['dataset_manifest_sha256']
    if trained == full_manifest_sha:
        return trained
    lines = (data / 'manifest.jsonl').read_bytes().splitlines(keepends=True)
    subset = [line for line in lines if json.loads(line)['split'] in ('train', 'val')]
    if len(subset) != 750 or len(lines) != 1000 or hashlib.sha256(b''.join(subset)).hexdigest() != trained:
        raise ValueError('Model training manifest differs from the exact local train/validation subset')
    return trained


def load_sequences(data, split='test'):
    path = data / 'manifest.jsonl'
    manifest_sha = digest(path)
    qa = read(data / 'QA.json')
    if qa['manifest_sha256'] != manifest_sha:
        raise ValueError('Dataset manifest does not match render QA')
    groups = defaultdict(list)
    ids = set()
    for line in path.read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        if set(row['camera_xy_target']) != set(TARGETS) or row['sample_id'] in ids:
            raise ValueError('Expected unique frames with camera optical-center XY labels')
        ids.add(row['sample_id'])
        groups[row['trajectory_id']].append(row)
    sequences = []
    for rows in groups.values():
        if len({row['split'] for row in rows}) != 1:
            raise ValueError('A trajectory occurs in multiple splits')
        if rows[0]['split'] != split:
            continue
        rows.sort(key=lambda row: row['frame_index'])
        if any(a['timestamp_s'] >= b['timestamp_s'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Causal sequences require increasing timestamps')
        frames = []
        for row in rows:
            image_path = (data / row['image_path']).resolve()
            if not image_path.is_relative_to(data.resolve()) or digest(image_path) != row['image_sha256']:
                raise ValueError('RGB differs from the frozen manifest')
            with Image.open(image_path) as image:
                frames.append(pixels(image).astype(np.float32) / 255)
        sequences.append((rows, frames))
    if not sequences:
        raise ValueError('Expected a nonempty sequence split')
    return sequences, manifest_sha, qa['field_mask_sha256']


def verified_graph(models, collection, setting, manifest_sha):
    route, window = SETTINGS[setting]
    members = {entry['name']: entry for entry in collection['files'] if entry['setting'] == setting}
    folder = models / setting
    for name in ('best.pt', 'camera_xy.onnx', 'METRICS.json', 'history.json'):
        path = folder / name
        member = members[f'models/{setting}/{name}']
        if member['bytes'] != path.stat().st_size or member['sha256'] != digest(path):
            raise ValueError(f'Saved Kaggle member differs: {setting}/{name}')
    checkpoint = torch.load(folder / 'best.pt', weights_only=True, map_location='cpu')
    if (checkpoint['route'], checkpoint['window'], tuple(checkpoint['targets']),
            checkpoint['dataset_manifest_sha256']) != (route, window, TARGETS, manifest_sha):
        raise ValueError(f'Dataset or camera-XY graph contract differs: {setting}')
    metrics = read(folder / 'METRICS.json')
    history = read(folder / 'history.json')
    if (metrics['selected_epoch'] <= 0 or metrics['optimizer_updates'] <= 0
            or metrics['optimizer_updates'] != history[-1]['optimizer_updates']
            or metrics['best_sha256'] != digest(folder / 'best.pt')
            or metrics['onnx_sha256'] != digest(folder / 'camera_xy.onnx')):
        raise ValueError(f'Training evidence differs: {setting}')
    centers = np.asarray(checkpoint['centers_mm'], dtype=float)
    axis = np.arange(-1175, 1200, 50)
    x, y = np.meshgrid(axis, axis)
    if not np.array_equal(centers, np.column_stack((x.ravel(), y.ravel()))):
        raise ValueError('Expected the corrected field grid in increasing y/x order')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(folder / 'camera_xy.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    model_input, = session.get_inputs()
    model_output, = session.get_outputs()
    if (model_input.type != 'tensor(float)' or model_input.shape != [1, 3 * window, 96, 160]
            or model_output.shape != [1, len(centers)] or model_output.name != 'location_logits'):
        raise ValueError(f'Unexpected causal graph interface: {setting}')
    return session, model_input.name, centers, metrics, checkpoint['parent_sha256'], members


def clip(frames, index, window, mode):
    indices = list(range(max(0, index - window + 1), index + 1))
    if mode == 'current_only':
        indices = [index] * window
    else:
        indices = [indices[0]] * (window - len(indices)) + indices
        if mode == 'past_reversed':
            indices = list(reversed(indices[:-1])) + [index]
    return np.concatenate([frames[i] for i in indices], axis=0)[None]


def score_grid(logits, centers, truth, temperature, width):
    score = np.exp((logits - logits.max()) / temperature)
    cell = 2400 / width
    indices = np.floor((centers + 1200) / cell).astype(int)
    grid = np.bincount(indices[:, 1] * width + indices[:, 0], weights=score, minlength=width**2)
    flat = int(grid.argmax())
    y, x = divmod(flat, width)
    position = np.array([(x + .5) * cell - 1200, (y + .5) * cell - 1200])
    return position, float(np.linalg.norm(position - truth)), float(grid[flat] / grid.sum())


def statistics(records):
    errors = np.array([record['error_mm'] for record in records])
    confidence = np.array([record['confidence'] for record in records])
    return dict(frames=len(records), median_mm=float(np.median(errors)), p95_mm=float(np.quantile(errors, .95)),
                mean_mm=float(errors.mean()), within50_rate=float((errors <= 50).mean()),
                brier_within50=float(np.mean((confidence - (errors <= 50))**2)),
                jumps_over500=sum(record['jump_mm'] > 500 for record in records))


def evaluate(sequences, models, collection, setting, manifest_sha):
    session, input_name, centers, calibration, parent, members = verified_graph(
        models, collection, setting, manifest_sha)
    _, window = SETTINGS[setting]
    temp = calibration['temperature']
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError(f'Invalid validation-calibrated temperature: {setting}')
    modes = ('chronological',) if window == 1 else ('chronological', 'current_only', 'past_reversed')
    result = {}
    for mode in modes:
        samples = {width: [] for width in (24, 48)}
        times = []
        for rows, frames in sequences:
            previous = {}
            for index, row in enumerate(rows):
                packed = clip(frames, index, window, mode)
                start = time.perf_counter()
                logits, = session.run(['location_logits'], {input_name: packed})
                times.append((time.perf_counter() - start) * 1000)
                if not np.isfinite(logits).all():
                    raise ValueError(f'Nonfinite location logits: {setting}')
                target = np.array([row['camera_xy_target'][key] for key in TARGETS])
                for width, records in samples.items():
                    xy, error, probability = score_grid(logits[0], centers, target, temp, width)
                    jump = float(np.linalg.norm(xy - previous[width])) if width in previous else 0.
                    previous[width] = xy
                    records.append(dict(frame_index=index, error_mm=error, confidence=probability, jump_mm=jump))
        result[mode] = {width: dict(all_endpoints=statistics(records),
                                    common_warm_endpoints=statistics([r for r in records if r['frame_index'] >= 7]),
                                    fully_observed_window=statistics([r for r in records if r['frame_index'] >= window - 1]))
                        for width, records in samples.items()}
        result[mode]['host_onnx_p95_ms'] = float(np.quantile(times, .95))
    return dict(parent_checkpoint_sha256=parent, selected_epoch=calibration['selected_epoch'],
                optimizer_updates=calibration['optimizer_updates'], temperature=temp, members=members,
                window=window, results=result)


def run(data, models, out, settings_to_run=SETTINGS):
    collection = read(models / 'MODEL_COLLECTION.json')
    if collection['status'] != 'ORIGINAL_NAMED_MODEL_MEMBERS_SHA256_PASS':
        raise ValueError('Expected original named Kaggle model-member verification')
    source = collection['source']
    if not source['ref'].startswith('/code/') or source['versionNumber'] < 1 or source['kernelId'] < 1:
        raise ValueError('Expected an exact saved Kaggle version')
    torch.set_num_threads(2)
    sequences, manifest_sha, field_sha = load_sequences(data)
    settings = {}
    for name in settings_to_run:
        trained = model_manifest_identity(data, models, name, manifest_sha)
        settings[name] = evaluate(sequences, models, collection, name, trained)
        settings[name]['training_manifest_sha256'] = trained
        print(name, settings[name]['results']['chronological'][48]['all_endpoints'], flush=True)
    if len({setting['parent_checkpoint_sha256'] for setting in settings.values()}) != 1:
        raise ValueError('Comparison mixed pretrained parents')
    report = dict(status='NEURAL_ONNX_CAUSAL_ABLATION_COMPLETE', dataset_manifest_sha256=manifest_sha,
                  corrected_field_mask_sha256=field_sha, saved_version=source, training_run_id=collection['run_id'],
                  frames=sum(len(rows) for rows, _ in sequences), trajectories=len(sequences),
                  settings=settings, host_runtime=ort.__version__, native_k230='NOT_RUN',
                  source_sha256=digest(Path(__file__)))
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    args = parser.parse_args()
    run(args.data, args.models, args.out, args.settings)
