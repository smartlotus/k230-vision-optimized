"""Continue an accepted formal run into model conversion and verified collection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from accept_full_downstream import accept_formal, formal_evidence
from cloud_store import atomic_json, sha
from collect_model_files import collect as collect_models
from collect_model_reports import collect as collect_reports
from prepare_model_artifact import prepare
from run_prepared_job import run as run_prepared_job


ROOT = Path(__file__).resolve().parents[2]
CLOUD = ROOT / 'docs/topface/v11/cloud'
TRAINING = CLOUD.parent / 'training'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def accepted_parent(frozen, collected, publication):
    acceptance = frozen / 'ACCEPTANCE.json'
    selected = frozen / 'SELECTED_MODEL.json'
    if not acceptance.exists() and not selected.exists():
        accept_formal(frozen, collected, publication, acceptance, selected)
    else:
        # A resumed controller must bind to the same published training session.
        intent, result, durable, _, trace, _ = formal_evidence(frozen, collected, publication)
        report, parent = read(acceptance), read(selected)
        best, = (row for row in trace['checkpoints'] if row['name'] == 'best.pt')
        if (result['status'] != 'FINITE_TRAINING_COMPLETE' or result['completed_epochs'] < 100 or
                report['run_id'] != intent['run_id'] or
                report['publication_receipt_sha256'] != sha(publication) or
                parent['kind'] != 'formal_model' or parent['publication'] != durable or
                parent['checkpoint_sha256'] != best['sha256']):
            raise ValueError('Existing formal acceptance differs from the original saved run')
    return selected


def continue_conversion(formal_revision, conversion_revision, token_file, source_commit, native_runtime):
    if not all(re.fullmatch(r'[a-z0-9]+', revision) for revision in (formal_revision, conversion_revision)):
        raise ValueError('Use explicit lowercase formal and conversion revisions')
    frozen = CLOUD / f'formal-{formal_revision}'
    intent = read(frozen / 'SUBMISSION_INTENT.json')
    run_id = intent['run_id']
    version = read(frozen / 'KERNEL_SAVE_RESPONSE.json')['versionNumber']
    collected = frozen / f'compact-v{version}'
    result = collected / f'{run_id}-RESULT.json'
    publication = CLOUD / 'publisher' / run_id / 'TRANSFER_RECEIPT.json'

    local = ROOT / '.ops/local'
    conversion = CLOUD / f'conversion-formal-{conversion_revision}'
    staging = local / f'topface-conversion-formal-{conversion_revision}-input'
    lock = local / f'topface-conversion-formal-{conversion_revision}.continuation.lock'
    progress = local / f'topface-conversion-formal-{conversion_revision}-continuation.json'

    state = {'formal_run_id': run_id, 'formal_version': version, 'pid': os.getpid()}

    def stage(name, **details):
        state.update(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(progress, state)
        print(json.dumps(state), flush=True)

    with lock.open('x', encoding='utf8') as stream:
        stream.write(str(os.getpid()))
    try:
        stage('WAITING_FOR_FORMAL_PUBLICATION')
        deadline = time.monotonic() + 13 * 3600
        while not publication.is_file():
            host = read(frozen / 'HOST_PROGRESS.json')
            if host['status'] == 'NEEDS_RECONCILIATION':
                raise RuntimeError('Training controller requires reconciliation')
            if time.monotonic() >= deadline:
                raise TimeoutError('Saved training publication wait expired')
            time.sleep(60)

        stage('CHECKING_ORIGINAL_FORMAL_ARCHIVE')
        formal_result = read(result)
        if formal_result['status'] == 'FORMAL_SESSION_BOUNDARY':
            stage('FORMAL_CHECKPOINT_READY_FOR_RESUME', completed_epochs=formal_result['completed_epochs'])
            return
        parent = accepted_parent(frozen, collected, publication)
        dataset_acceptance = TRAINING / 'FULL_DATASET_ACCEPTANCE.json'
        shard_inventory = TRAINING / 'FULL_INVENTORY.json'
        dataset_digest = read(dataset_acceptance)['dataset_index_sha256']
        checkpoint_digest = read(parent)['checkpoint_sha256']

        stage('PREPARING_FORMAL_CONVERSION', checkpoint_sha256=checkpoint_digest)
        if not conversion.exists():
            prepare(
                phase='formal_conversion', revision=conversion_revision, staging=staging, parent=parent,
                runtime_receipt=CLOUD / 'publisher/k230-topface-v11-compiler-linux-a02/TRANSFER_RECEIPT.json',
                runtime_qualification=CLOUD / 'compiler-linux-a02/compact-v2/k230-topface-v11-compiler-linux-a02-RESULT.json',
                native_runtime=native_runtime, dataset_acceptance=dataset_acceptance, shard_inventory=shard_inventory,
            )
        config = read(conversion / 'MODEL_ARTIFACT_CONFIG.json')
        if config['parent'] != read(parent) or config['dataset_index_sha256'] != dataset_digest:
            raise ValueError('Existing conversion binds a different checkpoint or evaluation dataset')

        conversion_id = config['run_id']
        receipt = CLOUD / 'publisher' / conversion_id / 'TRANSFER_RECEIPT.json'
        if not receipt.is_file():
            stage('RUNNING_FORMAL_CONVERSION', conversion_run_id=conversion_id)
            run_prepared_job(conversion, staging, token_file, source_commit)

        stage('COLLECTING_VERIFIED_MODELS_AND_REPORTS', conversion_run_id=conversion_id)
        saved_version = read(conversion / 'KERNEL_SAVE_RESPONSE.json')['versionNumber']
        compact = conversion / f'compact-v{saved_version}'
        saved = compact / f'{conversion_id}-RESULT.json'
        model_dir = local / f'topface-formal-{conversion_revision}-deliverables'
        collect_models(saved, receipt, token_file, model_dir, conversion / 'MODEL_FILES_RECEIPT.json')
        collect_reports(saved, receipt, token_file, compact / 'evaluation', conversion / 'REPORTS_RECEIPT.json')
        stage('CONVERSION_DELIVERABLES_VERIFIED', local_model_dir=str(model_dir),
              synthetic_report_dir=str(compact / 'evaluation'))
    except Exception as error:
        stage('NEEDS_RECONCILIATION', error_type=type(error).__name__)
        raise RuntimeError('Inspect the saved continuation stage before taking further action') from None
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-revision', required=True)
    parser.add_argument('--conversion-revision', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--native-runtime', type=Path, required=True)
    continue_conversion(**vars(parser.parse_args()))
