"""Publish the exact a06 A and TCN4 KModels with an isolated benchmark."""
import argparse
import json
from pathlib import Path

import torch

from dataset import digest
from package_offline import BOARD_DEPENDENCIES, write_package


def main():
    parser = argparse.ArgumentParser()
    for name in ('record', 'board-reference', 'parent', 'cnn', 'tcn', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    guide = source.parents[1] / 'docs/temporal_camera/map-correction/BOARD_MULTIROUTE.md'
    collection = json.loads((args.record / 'MODEL_COLLECTION.json').read_text())
    result = json.loads((args.record / 'compact/COMPARISON.json').read_text())
    saved = json.loads((args.record / 'KERNEL_SAVE_RESPONSE.json').read_text())
    if saved != collection['source'] or saved['kernelId'] != 145538534 or saved['versionNumber'] != 1:
        raise ValueError('Expected exact a06 model collection')
    intent = json.loads((args.record / 'SUBMISSION_INTENT.json').read_text())
    if digest(args.parent) != intent['parent_sha256']:
        raise ValueError('Deployment anchors must come from the original parent')
    anchor_centers = torch.load(args.parent, map_location='cpu', weights_only=True)['centers_mm']
    hashes = {entry['name']: entry['sha256'] for entry in collection['files']}
    files = {'README.md': (guide, None)}
    generated = {}
    for setting, route, window, root in (('cnn1', 'cnn', 1, args.cnn), ('tcn4', 'tcn', 4, args.tcn)):
        prefix = 'k230_temporal_v1_gray_' + setting + '/'
        report_path = root / 'KMODEL_REPORT.json'
        report = json.loads(report_path.read_text())
        model = result[setting]
        if (report['status'] != 'COMPILE_AND_SIMULATOR_PASS' or report['route'] != route
                or report['window'] != window or report['input_type'] != 'uint8'
                or report['checkpoint_sha256'] != hashes[f'models/{setting}/best.pt']
                or report['float_onnx_sha256'] != hashes[f'models/{setting}/camera_xy.onnx']
                or report['parent_sha256'] != intent['parent_sha256']):
            raise ValueError('Board model provenance differs from saved route')
        if json.loads((args.record / f'{setting.upper()}_KMODEL_REPORT.json').read_text()) != report:
            raise ValueError('Local compiled model report differs from the archived report')
        config = {
            'identity': {'window': window, 'input_rgb_uint8_nchw': report['input_shape'],
                         'map_centers_mm': anchor_centers},
            'temporal_filter': {'grid_size': 48, 'temperature': model['temperature'], 'memory_seconds': 2.,
                                'global_floor': .01, 'startup': 'competition', 'acceptance': None},
            'kmodel': {'kmodel_sha256': report['kmodel_sha256'], 'compiler': report['compiler']},
            'qualification': 'EXPERIMENTAL_ACCURACY_HOLD',
        }
        files[prefix + 'camera_xy.kmodel'] = (root / 'camera_xy.kmodel', report['kmodel_sha256'])
        files[prefix + 'KMODEL_REPORT.json'] = (report_path, None)
        files[prefix + 'board_helpers.py'] = (args.board_reference / 'board_helpers.py', BOARD_DEPENDENCIES['board_helpers.py'])
        for name in ('board_multiroute.py', 'board_temporal.py', 'competition_paths.py',
                     'embedded_filter.py', 'localization_runtime.py'):
            files[prefix + name] = (source / name, None)
        generated[prefix + 'deployment.json'] = (json.dumps(config, indent=2) + '\n').encode()
    write_package(args.out, files, generated, 'EXPERIMENTAL_ACCURACY_HOLD')


if __name__ == '__main__':
    main()
