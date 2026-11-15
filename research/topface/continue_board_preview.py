"""Wait for the original formal selection, then publish a separate early board package."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import zipfile

from cloud_store import atomic_json, sha
from accept_full_downstream import formal_evidence
from collect_model_files import collect
from package_board_preview import collect_preview_reports, package_preview
from prepare_board_preview import prepare_preview
from run_prepared_job import run


ROOT = Path(__file__).resolve().parents[2]
CLOUD = ROOT / 'docs/topface/v11/cloud'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def verify_original_selection(parent, accepted):
    expected = ('lotus30/k230-topface-v11-06-formal-training', 145144940, 1)
    provider = accepted['provider']
    if (accepted['run_id'] != 'k230-topface-v11-formal-a01' or
            parent['kind'] != 'formal_model' or parent['experiment'] != 'formal_full' or
            parent['completed_epochs'] < 100 or parent['completed_epochs'] != accepted['completed_epochs'] or
            parent['publication']['prep'] != provider or
            (provider['ref'], provider['kernel_id'], provider['version']) != expected or
            accepted['status'] != 'FORMAL_TRAINING_COMPUTE_AND_ARCHIVE_VERIFIED' or
            parent['checkpoint_sha256'] != accepted['selected_checkpoint']['sha256']):
        raise ValueError('Preview source must be the accepted original formal-a01 v1')


def continue_preview(token_file, source_commit, native_runtime, output):
    selected = CLOUD / 'formal-a01/SELECTED_MODEL.json'
    acceptance = CLOUD / 'formal-a01/ACCEPTANCE.json'
    local = ROOT / '.ops/local'
    progress = local / 'topface-formal-preview-a01-continuation.json'
    lock = local / 'topface-formal-preview-a01-continuation.lock'
    state = {'pid': os.getpid(), 'formal_ref': 'lotus30/k230-topface-v11-06-formal-training',
             'formal_version': 1, 'scope': 'FORMAL_BOARD_PREVIEW_ONLY', 'automatic_replay': False}
    def stage(name, **details):
        state.update(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(progress, state)
        print(json.dumps(state), flush=True)

    with lock.open('x', encoding='utf8') as stream:
        stream.write(str(os.getpid()))
    try:
        stage('WAITING_FOR_ORIGINAL_FORMAL_ACCEPTANCE')
        deadline = time.monotonic() + 6 * 3600
        while not selected.is_file() or not acceptance.is_file():
            host = read(CLOUD / 'formal-a01/HOST_PROGRESS.json')
            if host['status'] == 'NEEDS_RECONCILIATION':
                raise RuntimeError('Original formal session needs reconciliation')
            if time.monotonic() >= deadline:
                raise TimeoutError('Original model remains unavailable; do not replace training')
            time.sleep(30)
        parent, accepted = read(selected), read(acceptance)
        verify_original_selection(parent, accepted)
        formal_source = read(CLOUD / 'formal-a01/KERNEL_SAVE_RESPONSE.json')
        if formal_source['kernelId'] != 145144940 or formal_source['versionNumber'] != 1:
            raise ValueError('Expected the original formal v1')
        original_publication = CLOUD / 'publisher/k230-topface-v11-formal-a01/TRANSFER_RECEIPT.json'
        _, final_result, durable, _, trace, _ = formal_evidence(
            CLOUD / 'formal-a01', CLOUD / 'formal-a01/compact-v1', original_publication)
        if (parent['publication'] != durable or final_result['status'] != 'FINITE_TRAINING_COMPLETE' or
                accepted['publication_receipt_sha256'] != sha(original_publication) or
                parent['checkpoint_sha256'] != trace['best_checkpoint_sha256']):
            raise ValueError('Original formal evidence and selected preview parent differ')
        frozen = CLOUD / 'conversion-formal-previewa01'
        staging = local / 'topface-formal-preview-a01-input'
        stage('PREPARING_SEPARATE_CPU_PREVIEW', checkpoint_sha256=parent['checkpoint_sha256'])
        if not frozen.exists():
            prepare_preview('previewa01', staging, selected,
                            CLOUD / 'publisher/k230-topface-v11-compiler-linux-a02/TRANSFER_RECEIPT.json',
                            CLOUD / 'compiler-linux-a02/compact-v2/k230-topface-v11-compiler-linux-a02-RESULT.json',
                            native_runtime, CLOUD.parent / 'training/FULL_DATASET_ACCEPTANCE.json',
                            CLOUD.parent / 'training/FULL_INVENTORY.json')
        config, intent = read(frozen / 'MODEL_ARTIFACT_CONFIG.json'), read(frozen / 'SUBMISSION_INTENT.json')
        if (config['phase'] != 'formal_board_preview' or config['parent'] != parent or
                intent['kernel'] != 'lotus30/k230-topface-v11-07c-formal-board-preview' or
                intent['input_manifest_sha256'] != sha(staging / 'INPUT_MANIFEST.json')):
            raise ValueError('Preview identity differs from the separate accepted preparation')
        publication = CLOUD / 'publisher' / config['run_id'] / 'TRANSFER_RECEIPT.json'
        if not publication.is_file():
            stage('BUILDING_AND_PUBLISHING_SEPARATE_PREVIEW', preview_run_id=config['run_id'])
            run(frozen, staging, token_file, source_commit, poll_seconds=30)
        version = read(frozen / 'KERNEL_SAVE_RESPONSE.json')['versionNumber']
        result = frozen / f'compact-v{version}' / f"{config['run_id']}-RESULT.json"
        stage('COLLECTING_PREVIEW')
        collect(result, publication, token_file, local / 'topface-formal-preview-a01-models',
                frozen / 'MODEL_FILES_RECEIPT.json')
        collect_preview_reports(frozen, token_file)
        value = package_preview(frozen, publication, output)
        archive = output.parent / (output.name + '.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    bundle.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
        stage('FORMAL_BOARD_PREVIEW_READY', output=str(output.resolve()),
              package_inventory_sha256=sha(output / 'SHA256SUMS.json'),
              checkpoint_sha256=value['checkpoint_sha256'], full_934_test='PENDING_ORIGINAL_07B',
              zip_file=str(archive.resolve()), zip_bytes=archive.stat().st_size, zip_sha256=sha(archive))
    except Exception as error:
        stage('NEEDS_RECONCILIATION', error_type=type(error).__name__)
        raise RuntimeError('Inspect the original preview state before any retry') from None
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('token-file', 'native-runtime', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    continue_preview(**vars(parser.parse_args()))
