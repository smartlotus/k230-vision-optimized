"""Build a separate board preview while the original full assessment continues."""
import json
from pathlib import Path
import shutil
import subprocess
import time

from cloud_store import archive_directory, atomic_json, sha
from compiler_runtime import install
from downstream_job import mount_training_source
from formal_job import unpack_session
from full_dataset import load_shards
from native_job import verify_inputs


def run(input_root, runtime_input, run_id, training_input_root):
    input_root, runtime_input = Path(input_root).resolve(), Path(runtime_input).resolve()
    verify_inputs(input_root)
    verify_inputs(runtime_input)
    config = json.loads((input_root / 'MODEL_ARTIFACT_CONFIG.json').read_text(encoding='utf8'))
    if (config['run_id'] != run_id or config['phase'] != 'formal_board_preview' or
            config['parent']['kind'] != 'formal_model' or config['parent']['completed_epochs'] < 100):
        raise ValueError('Preview requires the completed, accepted formal model')
    work, mount = Path('/kaggle/working'), Path('/kaggle/input').resolve()
    scratch = (Path('/kaggle/temp') if Path('/kaggle/temp').exists() else Path('/tmp')) / run_id
    scratch.mkdir(exist_ok=False)
    output = work / run_id
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + config['max_seconds']
    state = {'run_id': run_id, 'status': 'STARTED', 'optimizer_updates': 0,
             'input_manifest_sha256': sha(input_root / 'INPUT_MANIFEST.json'),
             'scope': 'FORMAL_BOARD_PREVIEW_ONLY', 'full_934_test': 'PENDING_ORIGINAL_07B',
             'native_board_test': 'NOT_RUN', 'automatic_replay': False}
    try:
        from native_export_runtime import install as install_native
        state['native_export_dependencies'] = install_native(input_root, scratch / 'native-site')
        dataset = scratch / 'dataset'
        load_shards(config['shards'], mount, dataset, config['total_scenes'], config['data_seed'],
                    splits=('train',), deadline=deadline - 600)
        if sha(dataset / 'AUDIT_V11.json') != config['dataset_index_sha256']:
            raise ValueError('Preview train view differs from the accepted Full index')
        print('TOPFACE_PREVIEW_STAGE INPUTS_VERIFIED', flush=True)
        backgrounds, bodies = None, None
        training_input_root = Path(training_input_root).resolve()
        if sha(training_input_root / 'INPUT_MANIFEST.json') != config['training_input']['manifest_sha256']:
            raise ValueError('Preview enhancement inputs differ from formal training')
        verify_inputs(training_input_root)
        if config.get('ptq_background_bank') is not None:
            from background_augmentation import unpack
            bank = config['ptq_background_bank']
            backgrounds = unpack(training_input_root / 'real-background-bank.bin', scratch / 'backgrounds',
                                 bank['archive_sha256'], bank['manifest_sha256'])
        if config.get('ptq_body_negative_bank') is not None:
            from body_negative_bank import unpack as unpack_body
            bank = config['ptq_body_negative_bank']
            bodies = unpack_body(training_input_root / 'body-negative-bank.bin', scratch / 'bodies',
                                 bank['archive_sha256'], bank['manifest_sha256'])
        parent = config['parent']
        source = unpack_session(mount, scratch / 'parent', parent, parent['experiment'], ('best.pt',), deadline - 600)
        checkpoint = source / 'best.pt'
        if sha(checkpoint) != parent['checkpoint_sha256']:
            raise ValueError('Preview source differs from the selected formal weight')
        code = output / 'source'
        mount_training_source(input_root, code)
        for path in input_root.glob('*.py'):
            shutil.copy2(path, code / path.name)
        from export_trained_topface import export
        exported = output / 'exported'
        state['export'] = export(checkpoint, parent['checkpoint_sha256'], exported, dataset)
        if state['export']['status'] != 'PYTORCH_ONNX_PARITY_PASS':
            raise ValueError('Preview ONNX parity failed')
        python, environment = install(runtime_input, scratch / 'runtime')
        command = [str(python), str(input_root / 'compile_trained_topface.py'),
                   '--export-dir', str(exported), '--dataset', str(dataset),
                   '--output', str(output / 'compiled'), '--calibration-count', '64', '--pair-count', '12']
        if backgrounds is not None:
            command.extend(['--background-bank', str(backgrounds)])
        if bodies is not None:
            command.extend(['--body-negative-bank', str(bodies)])
        print('TOPFACE_PREVIEW_STAGE COMPILE_AND_TRAIN_PAIRING', flush=True)
        with (output / 'compile.log').open('wb') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(1800, max(1, int(deadline - time.monotonic() - 600))))
        conversion = json.loads((output / 'compiled/RESULT.json').read_text(encoding='utf8'))
        if conversion['status'] != 'TRAINED_MODEL_COMPILED_AND_PAIRED':
            raise ValueError('Preview compiler did not complete its paired contract')
        from package_trial import package
        package(output / 'compiled', exported, output / 'board-package')
        models = work / (run_id + '-models')
        models.mkdir()
        for path in (exported / 'topface.onnx', output / 'compiled/topface.kmodel',
                     output / 'compiled/identity.json', output / 'compiled/output_contract.json'):
            shutil.copy2(path, models / path.name)
        shutil.copy2(checkpoint, models / 'topface-best.pt')
        shutil.copy2(checkpoint, output / 'selected-best.pt')
        archive_directory(output / 'board-package', models / 'topface-board-bundle.tar.gz', deadline)
        state.update(status='TRAINED_EXPORT_COMPILE_AND_SIMULATOR_COMPLETE', identity=conversion['identity'],
                     paired_measurements=conversion['paired_measurements'],
                     model_files=[{'path': path.relative_to(work).as_posix(), 'bytes': path.stat().st_size,
                                   'sha256': sha(path)} for path in sorted(models.iterdir())],
                     evaluation_reports=[{'path': name, 'bytes': (output / name).stat().st_size,
                                          'sha256': sha(output / name)}
                                         for name in ('exported/RESULT.json', 'compiled/RESULT.json')])
    except Exception as error:
        state.update(status='MODEL_ARTIFACT_BUILD_FAILED', error_type=type(error).__name__)
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'STATE.json', state)
        try:
            archive = work / (run_id + '.tar.gz')
            archive_directory(output, archive, deadline)
            state.update(archive={'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive)},
                         durable_status='PUBLISHER_PENDING_EXACT_SAVED_OUTPUT')
        except Exception as error:
            state.update(durable_status='UNACCEPTED_KEEP_SAVED_OUTPUT', publication_error_type=type(error).__name__)
        atomic_json(work / (run_id + '-RESULT.json'), state)
        print('TOPFACE_PREVIEW_RESULT ' + json.dumps(state), flush=True)
    return state
