"""Freeze the geometry run, keeping the new test frames off the Kaggle inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


ROOT = Path(__file__).resolve().parent
DATASET = 'lotus30/k230-neural-camera-pilot-geo-a03'
KERNEL = 'lotus30/k230-neural-camera-xy-geometry-a03'
RUN = 'k230-neural-camera-xy-geometry-20261117-a03'
HISTORY_KERNEL = 'lotus30/k230-neural-camera-xy-history-a04'
HISTORY_RUN = 'k230-neural-camera-xy-history-20261117-a04'
LONG_KERNEL = 'lotus30/k230-neural-camera-xy-long-a05'
LONG_RUN = 'k230-neural-camera-xy-long-20261117-a05'
MIXTURE_KERNEL = 'lotus30/k230-neural-camera-xy-mixture-a06'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def training_manifest(manifest):
    data = manifest.read_bytes().splitlines(keepends=True)
    selected = [line for line in data if json.loads(line)['split'] in ('train', 'val')]
    rows = [json.loads(line) for line in selected]
    if len(rows) != 750 or len(data) != 1000 or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Expected a held-back test split and 750 unique training-side items')
    return b''.join(selected), rows


def add_bytes(archive, name, content):
    item = tarfile.TarInfo(name)
    item.size = len(content)
    archive.addfile(item, io.BytesIO(content))
    return dict(name=name, bytes=item.size, sha256=sha(content))


def prepare(data, audit_path, stage, record):
    audit = read(audit_path)
    manifest = data / 'manifest.jsonl'
    if audit['status'] != 'GEOMETRY_PILOT_AUDIT_PASS' or audit['manifest_sha256'] != digest(manifest):
        raise ValueError('Prepare only the exact audited geometry pilot')
    frozen, rows = training_manifest(manifest)
    stage.mkdir(parents=True, exist_ok=False)
    record.mkdir(parents=True, exist_ok=False)
    payload = stage / 'payload.bin'
    entries = []
    with tarfile.open(payload, 'x:gz') as tar:
        entries.append(add_bytes(tar, 'data/manifest.jsonl', frozen))
        for row in rows:
            file = data / row['image_path']
            value = file.read_bytes()
            if sha(value) != row['image_sha256']:
                raise ValueError('RGB member differs from the signed manifest: ' + file.name)
            entries.append(add_bytes(tar, 'data/' + row['image_path'], value))
        inventory = (json.dumps(entries, indent=2) + '\n').encode()
        add_bytes(tar, 'INPUT_MANIFEST.json', inventory)
    expected = {item['name']: item for item in entries}
    with tarfile.open(payload, 'r:gz') as tar:
        if set(tar.getnames()) != set(expected) | {'INPUT_MANIFEST.json'}:
            raise ValueError('Geometry cloud input archive differs from the training-side inventory')
        for name, item in expected.items():
            blob = tar.extractfile(name).read()
            if (len(blob), sha(blob)) != (item['bytes'], item['sha256']):
                raise ValueError('Geometry input archive byte readback failed')
    capsule = dict(payload_bytes=payload.stat().st_size, payload_sha256=digest(payload),
                   manifest_sha256=sha(inventory), training_manifest_sha256=sha(frozen),
                   full_local_manifest_sha256=digest(manifest), payload_members=len(entries) + 1,
                   expanded_bytes=sum(item['bytes'] for item in entries) + len(inventory),
                   exported_rgb=750, withheld_test_rgb=250, audit_sha256=digest(audit_path))
    write(stage / 'CAPSULE.json', capsule)
    write(record / 'CAPSULE.json', capsule)
    write(stage / 'dataset-metadata.json', dict(id=DATASET, title='K230 Neural Camera Pilot Geometry A03',
                                               licenses=[dict(name='CC0-1.0')]))
    prior = ROOT / 'cloud/joint-a01'
    old = read(prior / 'SUBMISSION_INTENT.json')
    frozen_notebook = prior / 'run.ipynb'
    if digest(frozen_notebook) != old['notebook_sha256']:
        raise ValueError('Original immutable input bootstrap changed')
    notebook = read(frozen_notebook)
    cell = notebook['cells'][0]
    bootstrap = ''.join(cell['source']).split('\ntraining_source = ', 1)[0]
    marker = "files=list(Path('/kaggle/input').rglob('payload.bin'))"
    if bootstrap.count(marker) != 1:
        raise ValueError('Cannot identify the reused parent/dependency bundle')
    original_bytes = 73711400
    bootstrap = bootstrap.replace(marker, marker + f'\nfiles=[f for f in files if f.stat().st_size=={original_bytes}]')
    main_source = (ROOT / 'train_neural.py').read_bytes()
    program = bootstrap + f"""
import shutil
payloads=[f for f in Path('/kaggle/input').rglob('payload.bin')
          if f.stat().st_size=={capsule['payload_bytes']} and hashlib.sha256(f.read_bytes()).hexdigest()=={capsule['payload_sha256']!r}]
assert len(payloads)==1
training=Path('/kaggle/working/geom_input')
training.mkdir(exist_ok=False)
with tarfile.open(payloads[0],'r:gz') as archive:
    members=archive.getmembers()
    assert len(members)=={capsule['payload_members']} and sum(member.size for member in members)=={capsule['expanded_bytes']}
    assert all(member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts for member in members)
    archive.extractall(training,filter='data')
manifest=training/'INPUT_MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=={capsule['manifest_sha256']!r}
for item in json.loads(manifest.read_text()):
    path=training/item['name']
    assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
shutil.copyfile(root/'parent.pt',training/'parent.pt')
training_source={main_source.decode('utf8')!r}
assert hashlib.sha256(training_source.encode()).hexdigest()=={sha(main_source)!r}
script=Path('/kaggle/working/train_neural.py')
script.write_text(training_source,encoding='utf8')
subprocess.run([sys.executable,str(script),'--input',str(training),'--run-id',{RUN!r},
                '--manifest-sha',{capsule['training_manifest_sha256']!r},'--parent-sha',{old['parent_sha256']!r},
                '--epochs','24','--recipe','clip_photometric','--train-and-val-only'],check=True,timeout=1800)
"""
    compile(program, 'geometry notebook', 'exec')
    cell.update(source=program.splitlines(keepends=True), outputs=[], execution_count=None)
    write(record / 'run.ipynb', notebook)
    metadata = read(prior / 'kernel-metadata.json')
    metadata.update(id=KERNEL, title='K230 Neural Camera XY Geometry A03',
                    dataset_sources=[f"{old['dataset']}/{old['dataset_version']}", f'{DATASET}/1'])
    write(record / 'kernel-metadata.json', metadata)
    write(record / 'SUBMISSION_INTENT.json',
          dict(run_id=RUN, kernel=KERNEL, dataset_versions=metadata['dataset_sources'], input=capsule,
               notebook_sha256=digest(record / 'run.ipynb'), source_sha256=sha(main_source),
               parent_sha256=old['parent_sha256'], epochs_max_each=24, seed=117, recipe='clip_photometric',
               provider_max_seconds=2100, automatic_retry=False))
    print(json.dumps(capsule))


def prepare_history(base, record):
    previous = read(base / 'SUBMISSION_INTENT.json')
    notebook_path = base / 'run.ipynb'
    if (previous['kernel'] != KERNEL or digest(notebook_path) != previous['notebook_sha256']
            or read(base / 'ARCHIVE_READBACK.json')['status'] != 'PRIVATE_KAGGLE_SAVED_ARCHIVE_FULL_BYTE_SHA256_PASS'):
        raise ValueError('A04 must start from the verified frozen A03 input contract')
    notebook = read(notebook_path)
    cell = notebook['cells'][0]
    program = ''.join(cell['source']).split('\ntraining_source=', 1)[0]
    source = (ROOT / 'train_neural.py').read_bytes()
    launch = f"""
training_source={source.decode('utf8')!r}
assert hashlib.sha256(training_source.encode()).hexdigest()=={sha(source)!r}
script=Path('/kaggle/working/train_neural.py')
script.write_text(training_source,encoding='utf8')
subprocess.run([sys.executable,str(script),'--input',str(training),'--run-id',{HISTORY_RUN!r},
                '--manifest-sha',{previous['input']['training_manifest_sha256']!r},'--parent-sha',{previous['parent_sha256']!r},
                '--epochs','24','--recipe','clip_photometric','--train-and-val-only',
                '--history-xy-weight','.35','--settings','tcn8'],check=True,timeout=1800)
"""
    compile(program + launch, 'history notebook', 'exec')
    cell.update(source=(program + launch).splitlines(keepends=True), outputs=[], execution_count=None)
    metadata = read(base / 'kernel-metadata.json')
    metadata.update(id=HISTORY_KERNEL, title='K230 Neural Camera XY History A04')
    record.mkdir(parents=True, exist_ok=False)
    write(record / 'run.ipynb', notebook)
    write(record / 'kernel-metadata.json', metadata)
    write(record / 'DATASET_CREATE_RESPONSE.json', read(base / 'DATASET_CREATE_RESPONSE.json'))
    previous.update(run_id=HISTORY_RUN, kernel=HISTORY_KERNEL,
                    notebook_sha256=digest(record / 'run.ipynb'), source_sha256=sha(source), history_xy_weight=.35,
                    settings=['tcn8'], prior_geometry_saved_version=read(base / 'KERNEL_SAVE_RESPONSE.json'))
    write(record / 'SUBMISSION_INTENT.json', previous)
    print(json.dumps(dict(history_xy_weight=.35, kernel=HISTORY_KERNEL,
                          exact_reused_dataset_versions=metadata['dataset_sources'])))


def submit(record, support):
    sys.path.insert(0, str(support.resolve()))
    from cloud import authenticate, network
    from kaggle.api.kaggle_api_extended import KaggleApi

    network()
    api = KaggleApi()
    authenticate(api)
    intent = read(record / 'SUBMISSION_INTENT.json')
    metadata = read(record / 'kernel-metadata.json')
    source = record / metadata['code_file']
    receipt = read(record / 'DATASET_CREATE_RESPONSE.json')
    attempt = record / 'KERNEL_SAVE_ATTEMPT.json'
    if (attempt.exists() or digest(source) != intent['notebook_sha256']
            or metadata['dataset_sources'] != intent['dataset_versions'] or not metadata['is_private']
            or metadata['id'] != intent['kernel'] or intent['kernel'] not in (KERNEL, HISTORY_KERNEL, LONG_KERNEL, MIXTURE_KERNEL)):
        raise ValueError('Reconcile prior attempt or frozen Save identity')
    if (not receipt['confirmation'] or not receipt['private'] or receipt['ref'] != DATASET
            or receipt['input_manifest_sha256'] != intent['input']['manifest_sha256']):
        raise ValueError('Geometry dataset receipt differs from frozen intent')
    statuses = {}
    for item in intent['dataset_versions']:
        name, version = item.rsplit('/', 1)
        status = api.dataset_status(name, format='json(status,current_version_number)')
        status = json.loads(status) if isinstance(status, str) else status
        if status.get('status') != 'ready' or status.get('current_version_number') != int(version):
            raise ValueError('Required pinned private dataset version is unavailable: ' + name)
        statuses[item] = status
    write(record / 'DATASET_READY.json', statuses)
    kernel = intent['kernel']
    candidates = api.kernels_list(mine=True, search=kernel.split('/')[1])
    if any(getattr(item, 'ref', None) == kernel for item in candidates):
        raise ValueError('Cloud notebook already exists, reconcile it before Save')
    with attempt.open('x', encoding='utf8') as stream:
        json.dump(dict(operation='SAVE_AND_RUN_ALL', kernel=kernel, source_sha256=digest(source),
                       automatic_retry=False, utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
    try:
        saved = api.kernels_push(str(record.resolve()), acc='NvidiaTeslaT4',
                                 timeout=str(intent['provider_max_seconds']), no_run=False)
    except Exception as error:
        write(record / 'KERNEL_SAVE_UNCERTAIN.json', dict(automatic_retry=False, error_type=type(error).__name__))
        raise RuntimeError('Reconcile original cloud notebook Save') from None
    value = type(saved).to_dict(saved)
    write(record / 'KERNEL_SAVE_RESPONSE.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='mode', required=True)
    for command, names in (('prepare', ('data', 'audit', 'stage', 'record')), ('submit', ('record', 'support')),
                           ('prepare-history', ('base', 'record'))):
        sub = commands.add_parser(command)
        for name in names:
            if name == 'support':
                sub.add_argument('--support', type=Path, default=ROOT / 'support')
            else:
                sub.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.data, args.audit, args.stage, args.record)
    elif args.mode == 'prepare-history':
        prepare_history(args.base, args.record)
    else:
        submit(args.record, args.support)
