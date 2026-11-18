"""Verify and archive the corrected-map fallback and earlier Kaggle comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import torch

from dataset import digest


BOARD_DEPENDENCIES = {
    'board_probe.py': '80d58f4466045375bd6e2917b23d8649cb5246bbc1b898c40da3f3482724c048',
    'board_helpers.py': '0fdb8b81c7a988bd8d39bc1f5fbbfb8aa7699905f6c85fe5a1aaeaf0fe01fd9f',
}
SETTINGS = {'cnn1', 'gru4', 'gru8', 'tcn4', 'tcn8'}
RESULT_FILES = {'best.pt', 'camera_xy.onnx', 'history.json', 'METRICS.json'}


def model_members(model_root, prefix):
    collection_path = model_root / 'MODEL_COLLECTION.json'
    collection = json.loads(collection_path.read_text())
    expected = {f'models/{setting}/{name}' for setting in SETTINGS for name in RESULT_FILES}
    if {entry['name'] for entry in collection['files']} != expected or len(collection['files']) != len(expected):
        raise ValueError('Incomplete five-setting Kaggle model collection')
    files = {prefix + 'MODEL_COLLECTION.json': (collection_path, None)}
    for entry in collection['files']:
        subpath = entry['name'].removeprefix('models/')
        local = model_root / subpath
        if local.stat().st_size != entry['bytes']:
            raise ValueError('Unexpected Kaggle model size')
        files[prefix + subpath] = (local, entry['sha256'])
    return files, collection


def write_package(out, files, generated, qualification):
    receipt_path = out.with_suffix('.SHA256.json')
    if out.exists() or receipt_path.exists():
        raise FileExistsError(out)
    inventory = {}
    for member, (path, expected_hash) in files.items():
        sha = digest(path)
        if expected_hash and sha != expected_hash:
            raise ValueError('Unexpected input SHA256 for ' + member)
        inventory[member] = dict(sha256=sha, bytes=path.stat().st_size)
    for member, data in generated.items():
        if member in inventory:
            raise ValueError('Duplicate member ' + member)
        inventory[member] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix('.partial')
    with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for member, (path, _) in files.items():
            bundle.write(path, member)
        for member, data in generated.items():
            bundle.writestr(member, data)
        bundle.writestr('SHA256_MANIFEST.json', json.dumps(inventory, indent=2) + '\n')
    with zipfile.ZipFile(temp) as bundle:
        if len(bundle.namelist()) != len(inventory) + 1 or set(bundle.namelist()) != set(inventory) | {'SHA256_MANIFEST.json'}:
            raise ValueError('Unexpected model archive members')
        for name, entry in inventory.items():
            with bundle.open(name) as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha != entry['sha256'] or bundle.getinfo(name).file_size != entry['bytes']:
                raise ValueError('Archive readback failed for ' + name)
    receipt = dict(archive=out.name, sha256=digest(temp), bytes=temp.stat().st_size,
                   verified_members=len(inventory), qualification=qualification)
    temp.rename(out)
    with receipt_path.open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser()
    for name in ('frozen', 'ptq', 'parent', 'comparison', 'board-reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.SHA256.json').exists():
        raise FileExistsError(args.out)
    source = Path(__file__).resolve().parent
    docs = source.parents[1] / 'docs/temporal_camera'
    report = json.loads((args.ptq / 'KMODEL_REPORT.json').read_text())
    if report['status'] != 'COMPILE_AND_SIMULATOR_PASS' or (
        report['route'], report['window'], report['input_type'], report['input_shape']
    ) != ('cnn', 1, 'uint8', [1, 3, 96, 160]):
        raise ValueError('Invalid corrected-map KModel result')
    if report['evaluated_frames'] != report['full_evaluation_split_frames']:
        raise ValueError('Expected complete validation split evidence')
    files = {
        'board/camera_xy.kmodel': (args.ptq / 'camera_xy.kmodel', report['kmodel_sha256']),
        'gray_frozen_fallback/parent.pt': (args.parent, report['parent_sha256']),
        'gray_frozen_fallback/best.pt': (args.frozen / 'best.pt', report['checkpoint_sha256']),
        'gray_frozen_fallback/camera_xy.onnx': (args.frozen / 'camera_xy.onnx', report['float_onnx_sha256']),
        'gray_frozen_fallback/camera_xy_uint8.onnx': (args.ptq / 'camera_xy_uint8.onnx', report['uint8_onnx_sha256']),
    }
    files.update({'board/' + name: (args.board_reference / name, sha)
                  for name, sha in BOARD_DEPENDENCIES.items()})
    for name in ('board_temporal.py', 'competition_paths.py', 'embedded_filter.py', 'localization_runtime.py'):
        files['board/' + name] = (source / name, None)
    for name in ('KMODEL_REPORT.json', 'SIMULATOR_SCORES.npz'):
        files['gray_frozen_fallback/' + name] = (args.ptq / name, None)
    files['gray_frozen_fallback/VAL_REPLAY_FLOAT.json'] = (docs / 'map-correction/VAL_REPLAY.json', None)
    files['gray_frozen_fallback/VALIDATION_COMPARISON.json'] = (args.ptq / 'VALIDATION_COMPARISON.json', None)
    files['README.md'] = (docs / 'map-correction/OFFLINE_MODEL_DELIVERY.md', None)
    model_files, _ = model_members(args.comparison, 'earlier_kaggle_a05/')
    files.update(model_files)
    for name in ('COMPARISON.json', 'PILOT_GATE.json', 'COMPUTE.json', 'RESULT.json'):
        files['earlier_kaggle_a05/' + name] = (docs / 'cloud/map-reuse-pilot-a05/compact' / name, None)

    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    centers = parent['centers_mm']
    config = {
        'identity': {'input_rgb_uint8_nchw': report['input_shape'], 'map_centers_mm': centers,
                     'confidence': None, 'candidate_set_confidence': None},
        'temporal_filter': {'grid_size': 48, 'temperature': 1., 'memory_seconds': 2.,
                            'global_floor': .01, 'startup': 'competition', 'acceptance': None},
        'kmodel': {'kmodel_sha256': report['kmodel_sha256'], 'compiler': report['compiler']},
        'qualification': 'EXPERIMENTAL_ACCURACY_HOLD',
        'native_k230': 'NOT RUN',
    }
    board_config = (json.dumps(config, indent=2) + '\n').encode()
    write_package(args.out, files, {'board/deployment.json': board_config}, config['qualification'])


if __name__ == '__main__':
    main()
