"""Run one bounded Kaggle session of the single formal training experiment."""
import json
import os
from pathlib import Path
import shutil
import time

from cloud_artifacts import extract_verified_archive
from cloud_store import archive_directory, atomic_json, sha
from downstream_job import mount_training_source
from full_dataset import load_shards
from native_job import verify_inputs


def unpack_session(mount_root, target, record, directory, wanted, deadline):
    names = {f'{directory}/{name}' for name in wanted}
    archive = record['archive']
    if record['publication']['status'] != 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED':
        raise ValueError('Formal parent archive must be verified on Drive')
    if any(archive[key] != record['publication']['uploaded'][key] for key in ('bytes', 'sha256')):
        raise ValueError('Formal parent archive differs from the Drive receipt')
    paths = list(mount_root.rglob(archive['name']))
    if len(paths) != 1:
        raise ValueError('Expected one mounted formal parent archive')
    path = paths[0].resolve()
    if (not path.is_relative_to(mount_root) or path.stat().st_size != archive['bytes']
            or sha(path, deadline) != archive['sha256']):
        raise ValueError('Mounted formal parent archive differs')
    extract_verified_archive(path, target, select=lambda *_: names)
    return target/directory


def run(input_root, run_id, max_seconds):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    work = Path('/kaggle/working')
    mount = Path('/kaggle/input').resolve()
    scratch = Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')
    if not work.is_dir() or not mount.is_dir():
        raise ValueError('Formal session requires Kaggle input and output mounts')
    input_root = Path(input_root).resolve()
    verify_inputs(input_root)
    config = json.loads((input_root/'FORMAL_CONFIG.json').read_text(encoding='utf8'))
    if config['run_id'] != run_id or config['max_seconds'] != max_seconds:
        raise ValueError('Formal session identity differs from the notebook')
    immutable = config['experiment']
    if config['dataset_acceptance']['status'] != 'FULL_DATASET_ACCEPTED':
        raise ValueError('Full dataset has not passed the independent audit')
    if config['pilot_admission']['full_render_admission'] != 'PASS':
        raise ValueError('Formal experiment requires a passing Pilot')

    root = scratch/run_id
    root.mkdir(exist_ok=False)
    output = root/'result'
    output.mkdir()
    started = time.monotonic()
    deadline = started + max_seconds
    state = {'schema': 'topface-formal-session-v1.1', 'run_id': run_id,
             'experiment_id': immutable['experiment_id'], 'session': config['session'],
             'status': 'STARTED', 'optimizer_updates': 0,
             'input_manifest_sha256': sha(input_root/'INPUT_MANIFEST.json')}

    def stage(name):
        state['stage'] = name
        print('TOPFACE_FORMAL_STAGE', json.dumps({'run_id': run_id, 'stage': name,
              'elapsed_seconds': time.monotonic()-started}), flush=True)

    try:
        stage('VERIFY_FULL_DATA')
        dataset = root/'dataset'
        index = load_shards(immutable['shards'], mount, dataset, immutable['scenes'],
                            immutable['data_seed'], deadline=deadline-600)
        if sha(dataset/'AUDIT_V11.json') != config['dataset_acceptance']['dataset_index_sha256']:
            raise ValueError('Materialized Full dataset index differs from accepted audit')
        state['split_images'] = index['split_images']
        stage('VERIFY_PARENT_CHECKPOINT')
        parent = config['parent']
        resume = parent['kind'] == 'formal_session'
        directory = ('formal_full' if resume else parent['experiment'])
        wanted = ('last.pt', 'best.pt') if resume else ('best.pt',)
        source = unpack_session(mount, root/'parent', parent, directory, wanted, deadline-600)
        checkpoint_name = 'last.pt' if resume else 'best.pt'
        if sha(source/checkpoint_name, deadline-600) != parent['checkpoint_sha256']:
            raise ValueError('Accepted parent checkpoint digest differs')
        mount_training_source(input_root, root/'code')
        from train_experiment import train

        stage('FORMAL_TRAINING')
        kwargs = ({'resume_directory': source, 'resume_sha256': parent['checkpoint_sha256']} if resume else {
            'initial_checkpoint': source/'best.pt', 'initial_checkpoint_sha256': parent['checkpoint_sha256']})
        result = train(dataset, output/'formal_full', input_root/'nanodet-pretrained.bin',
                       kind='formal_full', width=immutable['width'], height=immutable['height'],
                       seed=immutable['training_seed'], batch_size=immutable['batch_size'],
                       max_epochs=immutable['max_epochs'], deadline=deadline-600,
                       formal_identity=immutable['experiment_id'], session_epochs=config['session_epochs'],
                       **kwargs)
        state.update({key: result[key] for key in (
            'status', 'completed_epochs', 'optimizer_updates', 'checkpoints', 'best_epoch',
            'best_validation_score', 'gate', 'schedule', 'runtime')})
        state['epoch_records'] = result['epoch_records'][-config['session_epochs']:]
        summary = work/run_id/'formal_full'
        summary.mkdir(parents=True)
        shutil.copy2(output/'formal_full/BEST_METRICS.json', summary/'BEST_METRICS.json')
    except Exception as error:
        state.update(status='FAILED_PRESERVE_ORIGINAL_COMPUTE', error_type=type(error).__name__)
        result_file = output/'formal_full/RESULT.json'
        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding='utf8'))
            for key in ('optimizer_updates', 'completed_epochs', 'checkpoints'):
                if key in result:
                    state[key] = result[key]
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'STATE.json', state)
        try:
            archive = work/(run_id+'.tar.gz')
            archive_directory(output, archive, deadline)
            state['archive'] = {'name': archive.name, 'bytes': archive.stat().st_size,
                                'sha256': sha(archive, deadline)}
            state['durable_status'] = 'PUBLISHER_PENDING_EXACT_SAVED_OUTPUT'
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT',
                         publication_error_type=type(error).__name__)
        atomic_json(work/(run_id+'-RESULT.json'), state)
        print('TOPFACE_FORMAL_RESULT', json.dumps(state), flush=True)
    return state
