"""Freeze an additional CPU preview without editing the original conversion job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json, sha
from prepare_model_artifact import prepare, HERE, ROOT, read


def prepare_preview(revision, staging, parent, runtime_receipt, runtime_qualification,
                    native_runtime, dataset_acceptance, shard_inventory):
    if not revision.startswith('preview') or not revision.isalnum() or not revision[7:]:
        raise ValueError('Use a separate preview revision')
    dataset = 'lotus30/k230-topface-v11-board-preview-input-' + revision[7:]
    if not 6 <= len(dataset.split('/')[1]) <= 50:
        raise ValueError('Preview input slug exceeds the Kaggle CLI length contract')
    intent = prepare('formal_conversion', revision, staging, parent, runtime_receipt,
                     runtime_qualification, native_runtime, dataset_acceptance, shard_inventory)
    staging = staging.resolve()
    frozen = ROOT / f'docs/topface/v11/cloud/conversion-formal-{revision}'
    kernel = staging.with_name(staging.name + '-kernel')
    if any((frozen / name).exists() for name in ('DATASET_SAVE_ATTEMPT.json', 'KERNEL_SAVE_ATTEMPT.json')):
        raise ValueError('Do not alter any already submitted input')
    config = read(staging / 'MODEL_ARTIFACT_CONFIG.json')
    config.update(phase='formal_board_preview', max_seconds=5400, scope='FORMAL_BOARD_PREVIEW_ONLY',
                  full_934_test='PENDING_ORIGINAL_07B', native_board_test='NOT_RUN')
    atomic_json(staging / 'MODEL_ARTIFACT_CONFIG.json', config)
    shutil.copy2(HERE / 'model_board_preview_job.py', staging / 'model_board_preview_job.py')
    manifest = {'schema': 'topface-model-artifact-input-v1.1', 'run_id': config['run_id'],
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                          for path in sorted(staging.iterdir())
                          if path.name not in ('INPUT_MANIFEST.json', 'dataset-metadata.json')]}
    atomic_json(staging / 'INPUT_MANIFEST.json', manifest)
    metadata = read(kernel / 'kernel-metadata.json')
    original_input = intent['dataset'] + '/1'
    if metadata['dataset_sources'].count(original_input) != 1:
        raise ValueError('Expected one explicit private preview input binding')
    metadata['dataset_sources'] = [dataset + '/1' if value == original_input else value
                                   for value in metadata['dataset_sources']]
    data_metadata = read(staging / 'dataset-metadata.json')
    data_metadata['id'] = dataset
    atomic_json(staging / 'dataset-metadata.json', data_metadata)
    metadata.update(id='lotus30/k230-topface-v11-07c-formal-board-preview',
                    title='K230 Topface V11 07c Formal Board Preview')
    notebook = read(kernel / intent['notebook_file'])
    source = notebook['cells'][0]['source']
    source = source.replace(intent['input_manifest_sha256'], sha(staging / 'INPUT_MANIFEST.json'))
    source = source.replace('from model_artifact_job import run', 'from model_board_preview_job import run')
    if 'from model_board_preview_job import run' not in source:
        raise ValueError('Preview notebook worker binding failed')
    notebook['cells'][0]['source'] = source
    atomic_json(kernel / intent['notebook_file'], notebook)
    atomic_json(kernel / 'kernel-metadata.json', metadata)
    intent.update(kernel=metadata['id'], dataset=dataset, notebook_sha256=sha(kernel / intent['notebook_file']),
                  input_manifest_sha256=sha(staging / 'INPUT_MANIFEST.json'), provider_max_seconds=6300)
    for folder, names in ((staging, ('MODEL_ARTIFACT_CONFIG.json', 'INPUT_MANIFEST.json')),
                          (kernel, (intent['notebook_file'], 'kernel-metadata.json'))):
        for name in names:
            shutil.copy2(folder / name, frozen / name)
    atomic_json(frozen / 'SUBMISSION_INTENT.json', intent)
    return intent
