"""Paired replay of the saved a06 models and the time-scaled CPU prior."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from competition_paths import seed_start_belief
from dataset import audit, digest
from embedded_filter import SoftGridFilter
from map_models import MapCameraXYNet
from train_map_temporal import grid_prob, pixels, stats


PROCESS_SCALE_MM_S = 800.
GRID = 48


def first_visible(data, rows):
    for i, row in enumerate(rows):
        with Image.open(data / row['mask_path']) as mask:
            counts = mask.histogram()
        if any(counts[label] >= 15 for label in (1, 2, 4, 5, 6)):
            return i
    return None


def compare(rows, groups, scores, centers, temperature, visibility, oracle_start_area=False):
    all_results = {}
    scores = (scores - scores.max(axis=1, keepdims=True)) / temperature
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    likelihood = [grid_prob(row, centers, GRID).ravel().tolist() for row in probabilities]
    for name, scale in (('one_ring', None), ('time_scaled', PROCESS_SCALE_MM_S)):
        error = []
        latencies = []
        stable_delays = []
        routes = {}
        for group, ids in groups.items():
            filter_ = SoftGridFilter([[True] * GRID for _ in range(GRID)], cell_mm=2400 / GRID,
                                     memory_seconds=2., global_floor=.01, process_scale_mm_s=scale)
            seed_start_belief(filter_)
            if oracle_start_area:
                true_y = rows[ids[0]]['camera_xy_target']['camera_y_mm']
                filter_.initialize([p if (i // GRID >= GRID // 2) == (true_y > 0) else 0.
                                    for i, p in enumerate(filter_.belief)])
            values = []
            prediction_y0 = None
            for i in ids:
                row = rows[i]
                start = time.perf_counter()
                output = filter_.update(likelihood[i], row['timestamp_s'])
                latencies.append((time.perf_counter() - start) * 1000)
                target = row['camera_xy_target']
                values.append(math.hypot(output['camera_x_mm'] - target['camera_x_mm'],
                                         output['camera_y_mm'] - target['camera_y_mm']))
                if len(values) == 1:
                    prediction_y0 = output['camera_y_mm']
            observed = visibility[group]
            first = next((i for i in range(observed, len(values) - 2)
                         if max(values[i:i + 3]) <= 200), None) if observed is not None else None
            start_zone_right = prediction_y0 * rows[ids[0]]['camera_xy_target']['camera_y_mm'] > 0
            delay = rows[ids[first]]['timestamp_s'] - rows[ids[observed]]['timestamp_s'] if first is not None else None
            if delay is not None:
                stable_delays.append(delay)
            routes[group] = {'initial_zone_correct': start_zone_right, 'first_additional_class_frame': observed,
                             'first_stable_200mm_after_additional_class': first, 'delay_s': delay}
            error.extend(values)
        all_results[name] = {
            'error': stats(error), 'initial_zone_correct': sum(r['initial_zone_correct'] for r in routes.values()),
            'trials': len(routes), 'additional_class_visible': sum(i is not None for i in visibility.values()),
            'stable_three_frames_after_additional_class': len(stable_delays),
            'median_delay_s_on_stable_routes': float(np.median(stable_delays)) if stable_delays else None,
            'host_filter_p95_ms': float(np.quantile(latencies, .95)), 'trajectories': routes,
        }
    return all_results


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'parent', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--oracle-start-area', action='store_true',
                        help='Diagnostic only: provide ground-truth choice of the two areas at the first frame')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    manifest = args.data / 'manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    audit(rows, args.data)
    rows = [row for row in rows if row['split'] == 'test']
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['trajectory_id']].append(i)
    visibility = {group: first_visible(args.data, [rows[i] for i in ids]) for group, ids in groups.items()}
    checkpoint = torch.load(args.parent, map_location='cpu', weights_only=True)
    centers = np.asarray(checkpoint['centers_mm'])
    collection = json.loads((args.models / 'MODEL_COLLECTION.json').read_text())
    evidence = {entry['name']: entry['sha256'] for entry in collection['files']}
    parent_sha, manifest_sha = digest(args.parent), digest(manifest)
    images = torch.from_numpy(np.stack([pixels(args.data / row['image_path']) for row in rows]))
    result = {}
    with torch.inference_mode():
        for name in ('cnn1', 'gru4', 'gru8', 'tcn4', 'tcn8'):
            folder = args.models / name
            for item in ('best.pt', 'METRICS.json'):
                if digest(folder / item) != evidence[f'models/{name}/{item}']:
                    raise ValueError('Saved Kaggle file hash differs')
            saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            if saved['parent_sha256'] != parent_sha or saved['dataset_manifest_sha256'] != manifest_sha:
                raise ValueError('Replay does not match the saved training identities')
            metrics = json.loads((folder / 'METRICS.json').read_text())
            window = saved['window']
            model = MapCameraXYNet(checkpoint, saved['route'], window, saved['single_frame_adjustment']).eval()
            model.load_state_dict(saved['state'])
            features = torch.cat([model.image_embedding(batch) for batch in images.split(32)])
            windows = [None] * len(rows)
            for ids in groups.values():
                for offset, row_id in enumerate(ids):
                    use = ids[max(0, offset - window + 1):offset + 1]
                    windows[row_id] = [use[0]] * (window - len(use)) + use
            scores = torch.cat([model.score_features(batch) for batch in features[windows].split(32)]).numpy()
            result[name] = compare(rows, groups, scores, centers, metrics['temperature'], visibility,
                                   oracle_start_area=args.oracle_start_area)
            baseline = metrics['stress']['clean']['competition_two_zone/grid48']['filtered_grid']
            actual = result[name]['one_ring']['error']
            if actual['n'] != baseline['n']:
                raise ValueError('Baseline endpoints differ from the original saved metrics')
            if not args.oracle_start_area:
                result[name]['provider_baseline_delta_mm'] = {
                    key: actual[key] - baseline[key] for key in ('median_mm', 'p95_mm')
                }
            print(json.dumps({'setting': name, 'one_ring': actual,
                              'time_scaled': result[name]['time_scaled']['error'],
                              'provider_baseline_delta_mm': result[name].get('provider_baseline_delta_mm')}), flush=True)
    payload = {'status': 'PAIRED_TEST_REPLAY', 'trajectory_count': len(groups), 'frames': len(rows),
               'dataset_manifest_sha256': manifest_sha, 'provider_source': collection['source'],
               'process_scale_mm_s': PROCESS_SCALE_MM_S, 'scale_selection_split': 'val',
               'startup_policy': 'GT_START_AREA_DIAGNOSTIC_ONLY' if args.oracle_start_area else 'DUAL_BLIND',
               'conditions': 'same original clean trajectory endpoints and model parameters',
               'landmark_diagnostic': 'Masks are read only to annotate first additional class, not used as filter input',
               'native_k230': 'NOT RUN', 'settings': result}
    with args.out.open('x', encoding='utf8') as stream:
        json.dump(payload, stream, indent=2)


if __name__ == '__main__':
    main()
