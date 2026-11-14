"""Prepare a private Kaggle Full audit or formal training session."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

from cloud_store import atomic_json, sha
from prepare_full_shard import FOLDER_ID, SEED, TOTAL_SCENES
from nanodet_topface import CHECKPOINT_SHA256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def source_ref(receipt):
    prep = receipt['prep']
    if (receipt['status'], receipt['secret_cleanup'], receipt['folder_id'], prep['provider_status']) != (
        'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED', 'CONFIRMED_DELETED', FOLDER_ID, 'COMPLETE'
    ):
        raise ValueError('Private Kaggle source requires verified Drive publication')
    return f"{prep['ref']}/{prep['version']}"


def prepare(phase, revision, staging, shard_inventory, pilot_acceptance, publisher_receipt,
            dataset_acceptance=None, parent=None, session=1, session_epochs=20, max_epochs=100):
    if not revision.isalnum() or phase not in ('full-audit', 'formal'):
        raise ValueError('Explicit Full audit or formal revision required')
    pilot = read(pilot_acceptance)
    if pilot['full_render_admission'] != 'PASS':
        raise ValueError('Full plan requires a passing Pilot')
    source_ref(read(publisher_receipt))
    shards = sorted(read(shard_inventory)['shards'], key=lambda shard: shard['start_index'])
    position = 0
    for shard in shards:
        if shard['start_index'] != position or shard['count'] < 1:
            raise ValueError('Full shard ranges must be contiguous')
        position += shard['count']
    if position != TOTAL_SCENES:
        raise ValueError('Full dataset must cover the complete planned scene range')
    mounts = [source_ref(shard['publication']) for shard in shards]
    if len(set(mounts)) != len(mounts):
        raise ValueError('Full shard sources must be unique')
    run_id = f'k230-topface-v11-{phase}-{revision}'
    ordinal = '05' if phase == 'full-audit' else '06'
    kernel = f'lotus30/k230-topface-v11-{ordinal}-{phase}'
    dataset = f'lotus30/k230-topface-v11-{phase}-input-{revision}'
    files = ['cloud_store.py', 'cloud_artifacts.py', 'full_dataset.py', 'native_job.py',
             'scene_config.py', 'geometry.py', 'dataset_diagnostics.py', 'input_locator.py']
    if phase == 'full-audit':
        max_seconds = 5400
        module = 'full_audit_job'
        config = {'run_id': run_id, 'shards': shards, 'scenes': TOTAL_SCENES, 'seed': SEED}
        config_name = 'FULL_AUDIT_CONFIG.json'
        files += ['full_audit_job.py', 'near_similarity.py']
    else:
        max_seconds = 36000
        module = 'formal_job'
        config_name = 'FORMAL_CONFIG.json'
        if dataset_acceptance is None or parent is None:
            raise ValueError('Formal training needs an accepted Full audit and parent weights')
        audit = read(dataset_acceptance)
        previous = read(parent)
        if audit['status'] != 'FULL_DATASET_ACCEPTED' or audit['shard_inventory_sha256'] != sha(shard_inventory):
            raise ValueError('Formal shard inventory differs from the accepted Full audit')
        if not 100 <= max_epochs <= 200 or not 1 <= session_epochs <= max_epochs:
            raise ValueError('Formal logical and session epoch limits are inconsistent')
        if session < 1 or previous['kind'] != ('pilot' if session == 1 else 'formal_session'):
            raise ValueError('Formal parent must match the session order')
        if session > 1 and previous['session'] != session-1:
            raise ValueError('Formal resume must follow the previous accepted session')
        experiment = {'experiment_id': f'k230-topface-v11-formal-{revision}' if session == 1 else previous['experiment_id'],
                      'shards': shards, 'scenes': TOTAL_SCENES, 'data_seed': SEED,
                      'width': 320, 'height': 192, 'training_seed': 20261114,
                      'batch_size': 32, 'max_epochs': max_epochs,
                      'ellipse_shape_loss': pilot['selected_experiment']=='pilot_geometry_refine'}
        config = {'run_id': run_id, 'experiment': experiment, 'session': session,
                  'session_epochs': session_epochs, 'max_seconds': max_seconds, 'parent': previous,
                  'dataset_acceptance': audit, 'pilot_admission': pilot}
        mounts.append(source_ref(previous['publication']))
        files += ['formal_job.py', 'downstream_job.py', 'task_geometry.py', 'task_training.py',
                  'evaluate_task.py', 'train_experiment.py', 'nanodet_topface.py']
    staging = staging.resolve()
    kernel_dir = staging.with_name(staging.name+'-kernel')
    frozen = ROOT/'docs/topface/v11/cloud'/f'{phase}-{revision}'
    if any(path.exists() for path in (staging, kernel_dir, frozen)):
        raise FileExistsError('An existing preparation must be reconciled')
    staging.mkdir(parents=True)
    kernel_dir.mkdir()
    frozen.mkdir(parents=True)
    for name in files:
        shutil.copy2(HERE/name, staging/name)
    if phase == 'formal':
        checkpoint = ROOT/'.ops/local/topface-g3-models/nanodet-plus-m_320_checkpoint.ckpt'
        if sha(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError('Pretrained checkpoint identity changed')
        shutil.copy2(checkpoint, staging/'nanodet-pretrained.bin')
        with zipfile.ZipFile(staging/'nanodet-source.bin', 'w', zipfile.ZIP_DEFLATED) as bundle:
            for source in sorted((HERE/'vendor/nanodet_core').rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
                    bundle.write(source, source.relative_to(HERE).as_posix())
    atomic_json(staging/config_name, config)
    manifest = {'schema': 'topface-full-downstream-input-v1.1', 'run_id': run_id,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                          for path in sorted(staging.iterdir())]}
    atomic_json(staging/'INPUT_MANIFEST.json', manifest)
    manifest_sha = sha(staging/'INPUT_MANIFEST.json')
    title = f'K230 Topface V11 {ordinal} ' + ('Full Audit' if phase == 'full-audit' else 'Formal Training')
    atomic_json(staging/'dataset-metadata.json', {
        'id': dataset, 'title': title+' '+revision, 'licenses': [{'name': 'other'}],
        'description': 'Private input with verified source versions and configuration.',
    })
    name = f'{ordinal}_{phase}.ipynb'
    code = (HERE/'input_locator.py').read_text(encoding='utf8') + f'''
import sys
input_root = locate_input('/kaggle/input', {manifest_sha!r})
sys.path.insert(0, str(input_root))
from {module} import run
run(input_root, {run_id!r}, max_seconds={max_seconds})
'''
    atomic_json(kernel_dir/name, {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
        'cells': [{'cell_type': 'code', 'id': 'execute', 'metadata': {},
                   'source': code, 'execution_count': None, 'outputs': []}],
    })
    atomic_json(kernel_dir/'kernel-metadata.json', {
        'id': kernel, 'title': title, 'code_file': name, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': phase == 'formal', 'enable_internet': False,
        'dataset_sources': [dataset+'/1'], 'kernel_sources': mounts, 'competition_sources': [],
    })
    for origin, names in ((staging, ('INPUT_MANIFEST.json', config_name)),
                          (kernel_dir, (name, 'kernel-metadata.json'))):
        for filename in names:
            shutil.copy2(origin/filename, frozen/filename)
    intent = {'status': 'PREPARED_NOT_SUBMITTED', 'phase': phase, 'run_id': run_id,
              'kernel': kernel, 'dataset': dataset, 'dataset_version': 1,
              'notebook_file': name, 'notebook_sha256': sha(kernel_dir/name),
              'input_manifest_sha256': manifest_sha, 'provider_max_seconds': max_seconds+900,
              'execution_mode': 'API_SAVE_AND_RUN_ALL',
              'publisher_admission_status': 'QUALIFIED_AND_AUTHORIZED', 'automatic_replay': False}
    atomic_json(frozen/'SUBMISSION_INTENT.json', intent)
    return intent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('full-audit', 'formal'))
    parser.add_argument('--revision', required=True)
    for name in ('staging', 'shard-inventory', 'pilot-acceptance', 'publisher-receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('dataset-acceptance', 'parent'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--session-epochs', type=int, default=20)
    parser.add_argument('--max-epochs', type=int, default=100)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
