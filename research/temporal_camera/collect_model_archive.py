"""Collect exact named deployment files directly into a verified local ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from cloud import authenticate, network
from cloud_provider import read_output
from dataset import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    record = args.record.resolve()
    output = args.out.resolve()
    temp = output.with_suffix('.partial.zip')
    if output.exists() or output.with_suffix('.SHA256.json').exists():
        raise FileExistsError(output)
    if args.resume != temp.exists():
        raise ValueError('Use --resume only for the preserved partial archive from this exact run')
    intent = json.loads((record / 'SUBMISSION_INTENT.json').read_text())
    saved = json.loads((record / 'KERNEL_SAVE_RESPONSE.json').read_text())
    observed = json.loads((record / f"OBSERVATION-v{saved['versionNumber']}.json").read_text())
    collection = json.loads((record / 'MODEL_COLLECTION.json').read_text())
    if (observed['status'] != 'COMPLETE' or observed['kernel_id'] != saved['kernelId']
            or not observed['is_private'] or collection['source'] != saved or collection['run_id'] != intent['run_id']):
        raise ValueError('Expected the exact completed private saved version')
    expected = {entry['name']: entry for entry in collection['files']}
    required = {f'models/{setting}/{name}' for setting in ('cnn1', 'gru4', 'gru8', 'tcn4', 'tcn8')
                for name in ('best.pt', 'camera_xy.onnx', 'history.json', 'METRICS.json')}
    if set(expected) != required or len(collection['files']) != len(required):
        raise ValueError('Expected all 20 named original deployment files')
    inventory = {name: {'bytes': info['bytes'], 'sha256': info['sha256']} for name, info in expected.items()}
    for name in ('SUBMISSION_INTENT.json', 'KERNEL_SAVE_RESPONSE.json', 'MODEL_COLLECTION.json',
                 'REPORT.md', 'INPUT_MANIFEST.json', f"OBSERVATION-v{saved['versionNumber']}.json"):
        p = record / name
        inventory['run/' + name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    for item in json.loads((record / 'COLLECTION.json').read_text()):
        if item.get('available') is False:
            continue
        name = Path(item['name']).name
        inventory['run/compact/' + name] = {'bytes': item['bytes'], 'sha256': item['sha256']}
    network()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    authenticate(api)
    ref = saved['ref'].removeprefix('/code/')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temp, 'a' if args.resume else 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) - set(inventory):
            raise ValueError('Unexpected entries in the preserved partial archive')
        with api.build_kaggle_client() as client:
            remote_manifest = read_output(client, ref, saved['versionNumber'], intent['run_id'] + '/MEMBERS.json', 2 * 1024**2)
            remote = {item['name']: item for item in json.loads(remote_manifest)}
            for name, entry in inventory.items():
                if name in expected and (remote[name]['bytes'], remote[name]['sha256']) != (entry['bytes'], entry['sha256']):
                    raise ValueError('Remote member manifest differs from the verified named collection')
                if name in archive.namelist():
                    with archive.open(name) as stream:
                        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if archive.getinfo(name).file_size != entry['bytes'] or sha != entry['sha256']:
                        raise ValueError('Preserved partial member differs')
                    continue
                if name.startswith('models/'):
                    raw = read_output(client, ref, saved['versionNumber'], intent['run_id'] + '/' + name, 12 * 1024**2)
                else:
                    raw = (record / name.removeprefix('run/')).read_bytes()
                if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                    raise ValueError('Deployment archive member differs from saved identity')
                archive.writestr(name, raw)
            archive.writestr('SHA256_MANIFEST.json', json.dumps(inventory, indent=2) + '\n')
    with zipfile.ZipFile(temp) as archive:
        if set(archive.namelist()) != set(inventory) | {'SHA256_MANIFEST.json'}:
            raise ValueError('Delivered archive inventory differs')
        for name, info in inventory.items():
            with archive.open(name) as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha != info['sha256']:
                raise ValueError('Compressed archive member failed readback')
    receipt = {'archive': output.name, 'bytes': temp.stat().st_size, 'sha256': digest(temp),
               'verified_members': len(inventory), 'source': saved, 'qualification': 'EXPERIMENTAL_ACCURACY_HOLD'}
    temp.rename(output)
    with output.with_suffix('.SHA256.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
