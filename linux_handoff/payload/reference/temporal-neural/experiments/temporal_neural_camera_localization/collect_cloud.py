"""Reconcile an exact saved run and verify its complete bounded archive in memory."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def run(record, support):
    sys.path.insert(0, str(support.resolve()))
    from cloud import network, authenticate
    from cloud_provider import observe, read_output, persist_output
    from kaggle.api.kaggle_api_extended import KaggleApi

    read = lambda name: json.loads((record / name).read_text(encoding='utf8'))
    intent, saved = read('SUBMISSION_INTENT.json'), read('KERNEL_SAVE_RESPONSE.json')
    ref, version = saved['ref'].removeprefix('/code/'), saved['versionNumber']
    network()
    api = KaggleApi()
    authenticate(api)
    observe(api, ref, version, record / f'OBSERVATION-v{version}.json')
    status = read(f'OBSERVATION-v{version}.json')
    if (status['kernel_id'] != saved['kernelId'] or not status['is_private']
            or status['status'] != 'COMPLETE'):
        raise ValueError('Expected completed exact private saved version')
    with api.build_kaggle_client() as client:
        for name in ('RESULT.json', 'ENVIRONMENT.json'):
            persist_output(record / name, read_output(client, ref, version, name, 1 << 20))
        result = read('RESULT.json')
        if (result['run_id'] != intent['run_id'] or result['compute_status'] != 'PASS'
                or result['automatic_replay'] or result['train_exit_code'] != 0):
            raise ValueError('Expected original successful compute identity')
        archive = result['archive']
        raw = read_output(client, ref, version, archive['name'], 16 << 20)
    sha = lambda content: hashlib.sha256(content).hexdigest()
    if (len(raw), sha(raw)) != (archive['bytes'], archive['sha256']):
        raise ValueError('Original archive readback failed')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as package:
        files = {member.name: member for member in package if member.isfile()}
        root = result['run_id']
        manifest = json.load(package.extractfile(files.pop(f'{root}/MEMBERS.json')))
        if len(manifest) != len(files):
            raise ValueError('Unexpected saved archive members')
        for member in manifest:
            content = package.extractfile(files[f"{root}/{member['name']}"]).read()
            if (len(content), sha(content)) != (member['bytes'], member['sha256']):
                raise ValueError('Saved archive member differs from manifest')
    receipt = dict(status='PRIVATE_KAGGLE_SAVED_ARCHIVE_FULL_BYTE_SHA256_PASS', run_id=root,
                   source=saved, artifact=archive, verified_members=len(manifest) + 1,
                   drive_acceptance='DEFERRED', archive_saved_to_host=False, automatic_replay=False)
    persist_output(record / 'ARCHIVE_READBACK.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(receipt['status'], receipt['verified_members'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--support', type=Path, default=Path(__file__).resolve().parent / 'support')
    args = parser.parse_args()
    run(args.record, args.support)
