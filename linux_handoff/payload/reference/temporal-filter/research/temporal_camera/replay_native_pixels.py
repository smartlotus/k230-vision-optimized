"""Replay existing camera-free CanMV AI2D bytes through the a06 compiled models."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from dataset import digest


def main():
    ap = argparse.ArgumentParser()
    for name in ('evidence', 'cnn', 'tcn', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    native = args.evidence / 'final/native.json'
    prior = args.evidence / 'verification-final.json'
    receipt = json.loads(prior.read_text())
    raw_receipt = json.loads(native.read_text())
    native_hash = digest(native)
    if receipt['native_sha256'] != native_hash or raw_receipt['runtime'] != '2.11.0':
        raise ValueError('Prior native byte witness differs')
    observations = {}
    for record in receipt['preprocessing']:
        name = record['name']
        golden = args.evidence / 'inputs-after' / (name + '.golden')
        board = args.evidence / 'final' / (name + '.ai2d.rgb')
        if digest(golden) != record['golden_sha256'] or digest(board) != record['native_sha256']:
            raise ValueError('Prior offline or native bytes changed')
        pair = [np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(1, 3, 96, 160)
                for path in (golden, board)]
        observations[name] = pair
    plugin = str(Path(importlib.util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH'] = plugin
    os.environ['PATH'] = plugin + os.pathsep + os.environ.get('PATH', '')
    dll = os.add_dll_directory(plugin) if os.name == 'nt' else None
    import nncase
    if importlib.metadata.version('nncase') != '2.11.0':
        raise ValueError('Expected the same Simulator version used for compilation')
    out = args.out.resolve()
    models = [(args.cnn.resolve(), 1), (args.tcn.resolve(), 4)]
    os.chdir(out.parent)
    comparison = {'status': 'HISTORICAL_NATIVE_PIXELS_SIMULATOR_REPLAY', 'native_receipt_sha256': native_hash,
                  'historical_firmware': raw_receipt['firmware'], 'fixture_count': len(observations),
                  'scope': 'Replays native AI2D and host fixture pixels as exact binaries; no current acquisition or KPU',
                  'tcn_input': 'Each synthetic fixture repeated across all four slots', 'native_k230': 'NOT RUN',
                  'models': {}}
    for path, window in models:
        contract = json.loads((path / 'KMODEL_REPORT.json').read_text())
        compiled = path / 'camera_xy.kmodel'
        if contract['input_shape'] != [1, 3 * window, 96, 160] or digest(compiled) != contract['kmodel_sha256']:
            raise ValueError('Saved KModel input identity differs')
        model = nncase.Simulator()
        model.load_model(compiled.read_bytes())
        pairs = {}
        for name, (golden, native_pixels) in observations.items():
            pixel_diff = np.abs(golden.astype(np.int16) - native_pixels)
            scores = []
            for value in (golden, native_pixels):
                packed = np.tile(value, (1, window, 1, 1))
                model.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(packed))
                model.run()
                scores.append(model.get_output_tensor(0).to_numpy().copy())
            if any(x.shape != (1, 2304) or not np.isfinite(x).all() for x in scores):
                raise ValueError('Unexpected or nonfinite simulator logits')
            diff = np.abs(scores[0] - scores[1])
            pairs[name] = {'pixels_equal': bool(not pixel_diff.any()), 'pixel_mean_abs': float(pixel_diff.mean()),
                           'pixel_max_abs': int(pixel_diff.max()), 'logit_max_abs': float(diff.max()),
                           'logit_mean_abs': float(diff.mean()), 'golden_best_anchor': int(scores[0].argmax()),
                           'native_pixel_best_anchor': int(scores[1].argmax()),
                           'anchor_agreement': bool(scores[0].argmax() == scores[1].argmax())}
        comparison['models'][contract['route']] = {'window': window, 'kmodel_sha256': contract['kmodel_sha256'],
                                                   'fixtures': pairs}
    with out.open('x', encoding='utf8') as stream:
        json.dump(comparison, stream, indent=2)
    print(json.dumps(comparison))


if __name__ == '__main__':
    main()
