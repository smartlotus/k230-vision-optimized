"""Package the exact corrected-map saved models and the matching compiled graphs."""
import argparse
import json
from pathlib import Path

from package_offline import model_members, write_package


def main():
    parser = argparse.ArgumentParser()
    for name in ('record', 'models', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--compiled', nargs=2, action='append', required=True, metavar=('SETTING', 'DIRECTORY'))
    args = parser.parse_args()
    files, collection = model_members(args.models, 'models/')
    expected = json.loads((args.record / 'KERNEL_SAVE_RESPONSE.json').read_text())
    status_path = args.record / f"OBSERVATION-v{expected['versionNumber']}.json"
    status = json.loads(status_path.read_text())
    if collection['source'] != expected or status['status'] != 'COMPLETE' or not status['is_private']:
        raise ValueError('Saved model collection identity differs')
    intent = json.loads((args.record / 'SUBMISSION_INTENT.json').read_text())
    if collection['run_id'] != intent['run_id'] or status['kernel_id'] != expected['kernelId']:
        raise ValueError('Exact saved version required')
    files['parent.pt'] = (args.parent, intent['parent_sha256'])
    for name in ('REPORT.md', 'SUBMISSION_INTENT.json', 'INPUT_MANIFEST.json', status_path.name):
        files['run/' + name] = (args.record / name, None)
    for entry in json.loads((args.record / 'COLLECTION.json').read_text()):
        if entry.get('available') is False:
            continue
        name = Path(entry['name']).name
        files['run/' + name] = (args.record / 'compact' / name, entry['sha256'])
    hashes = {entry['name']: entry['sha256'] for entry in collection['files']}
    seen = set()
    for setting, directory in args.compiled:
        if setting not in ('cnn1', 'tcn4') or setting in seen:
            raise ValueError('Unexpected compiled setting ' + setting)
        seen.add(setting)
        path = Path(directory)
        report = json.loads((path / 'KMODEL_REPORT.json').read_text())
        if (report['status'] != 'COMPILE_AND_SIMULATOR_PASS'
                or report['kmodel_bytes'] <= 0
                or report['parent_sha256'] != intent['parent_sha256']
                or report['checkpoint_sha256'] != hashes[f'models/{setting}/best.pt']
                or report['float_onnx_sha256'] != hashes[f'models/{setting}/camera_xy.onnx']):
            raise ValueError('Compiled identity differs from saved training output')
        for name, key in (('camera_xy.kmodel', 'kmodel_sha256'), ('camera_xy_uint8.onnx', 'uint8_onnx_sha256')):
            files[f'compiled/{setting}/{name}'] = (path / name, report[key])
        files[f'compiled/{setting}/KMODEL_REPORT.json'] = (path / 'KMODEL_REPORT.json', None)
        files[f'compiled/{setting}/SIMULATOR_SCORES.npz'] = (path / 'SIMULATOR_SCORES.npz', None)
    write_package(args.out, files, {}, 'EXPERIMENTAL_ACCURACY_HOLD')


if __name__ == '__main__':
    main()
