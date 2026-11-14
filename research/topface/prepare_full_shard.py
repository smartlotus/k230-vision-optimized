"""Freeze one range of the V1.1 Full simulation after Pilot acceptance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json, sha


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FOLDER_ID = '1MwltNmaI24HjvcxnG1w608qui42RAPKf'
CAD_SHA256 = '642df0a28329ebe3adab0ca96888e5078b1a005dc7b9dbd345903e10274fc7f2'
TOTAL_SCENES = 10000
SHARD_SCENES = 1000
SEED = 20261117


def prepare(shard, staging, cad, pilot_admission, publisher_receipt):
    shard_count = TOTAL_SCENES // SHARD_SCENES
    if not 1 <= shard <= shard_count:
        raise ValueError(f'Shard must be between 1 and {shard_count}')
    pilot = json.loads(pilot_admission.read_text(encoding='utf8'))
    receipt = json.loads(publisher_receipt.read_text(encoding='utf8'))
    if pilot['full_render_admission'] != 'PASS':
        raise ValueError('Pilot training must accept Full rendering')
    if (receipt['status'], receipt['secret_cleanup'], receipt['folder_id']) != (
        'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED', 'CONFIRMED_DELETED', FOLDER_ID
    ):
        raise ValueError('Qualified Drive publisher required')
    if sha(cad) != CAD_SHA256:
        raise ValueError('Original CAD digest differs')

    revision = f'full-{shard:02d}-a01'
    run_id = f'k230-topface-v11-{revision}'
    dataset = f'lotus30/k230-topface-v11-full-input-{shard:02d}-a01'
    kernel = f'lotus30/k230-topface-v11-04-full-shard-{shard:02d}'
    start_index = (shard - 1) * SHARD_SCENES
    staging = staging.resolve()
    kernel_dir = staging.with_name(staging.name + '-kernel')
    frozen = ROOT / 'docs/topface/v11/cloud' / revision
    if any(path.exists() for path in (staging, kernel_dir, frozen)):
        raise FileExistsError('An existing preparation must be reconciled')
    staging.mkdir(parents=True)
    kernel_dir.mkdir()
    frozen.mkdir(parents=True)

    for name in ('geometry.py', 'cloud_store.py', 'native_job.py', 'native_cad.py',
                 'render_native.py', 'scene_config.py', 'audit_v11.py', 'coco_topface.py',
                 'input_locator.py'):
        shutil.copy2(HERE/name, staging/name)
    shutil.copy2(cad, staging/'cad_bundle.bin')
    smoke = ROOT/'docs/topface/v11/reports/SMOKE_HUMAN_REVIEW.json'
    shutil.copy2(smoke, staging/smoke.name)
    config = {
        'run_id': run_id, 'count': SHARD_SCENES, 'start_index': start_index,
        'seed': SEED, 'total_scenes': TOTAL_SCENES, 'pilot_admission': pilot,
        'publisher_receipt': receipt,
    }
    atomic_json(staging/'FULL_RENDER_CONFIG.json', config)
    manifest = {
        'schema': 'topface-full-render-input-v1.1',
        'run_id': run_id, 'global_scene_range': [start_index, start_index + SHARD_SCENES],
        'seed': SEED, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                  for path in sorted(staging.iterdir())],
    }
    atomic_json(staging/'INPUT_MANIFEST.json', manifest)
    manifest_sha = sha(staging/'INPUT_MANIFEST.json')
    atomic_json(staging/'dataset-metadata.json', {
        'id': dataset, 'title': f'K230 TopFace V1.1 Full input {shard:02d}',
        'licenses': [{'name': 'other'}],
        'description': 'Private V1.1 simulation code and original CAD for one fixed scene range.',
    })
    name = '04_full_shard.ipynb'
    code = (HERE/'input_locator.py').read_text(encoding='utf8') + f'''
import sys
input_root = locate_input('/kaggle/input', {manifest_sha!r})
sys.path.insert(0, str(input_root))
from native_job import main
main(input_root, '/kaggle/working', {run_id!r}, count={SHARD_SCENES},
     seed={SEED}, max_seconds=7200)
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
        'cells': [{'cell_type': 'markdown', 'id': 'contract', 'metadata': {},
                   'source': f'# Full V1.1 shard {shard}/{shard_count}\n\n'
                             f'Global scene indices {start_index}–{start_index + SHARD_SCENES - 1}. '
                             'Render and audit the original CAD with native Blender. '
                             'The verified archive is published to Drive after Kaggle saves it.'},
                  {'cell_type': 'code', 'id': 'render', 'metadata': {}, 'source': code,
                   'execution_count': None, 'outputs': []}]}
    atomic_json(kernel_dir/name, notebook)
    atomic_json(kernel_dir/'kernel-metadata.json', {
        'id': kernel, 'title': f'K230 Topface V11 04 Full Shard {shard:02d}',
        'code_file': name, 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [dataset+'/1'], 'kernel_sources': [], 'competition_sources': [],
    })
    for parent, files in ((staging, ('INPUT_MANIFEST.json', 'FULL_RENDER_CONFIG.json')),
                          (kernel_dir, (name, 'kernel-metadata.json'))):
        for file in files:
            shutil.copy2(parent/file, frozen/file)
    intent = {'status': 'PREPARED_NOT_SUBMITTED', 'phase': 'full_render', 'run_id': run_id,
              'kernel': kernel, 'dataset': dataset, 'dataset_version': 1,
              'notebook_file': name, 'notebook_sha256': sha(kernel_dir/name),
              'input_manifest_sha256': manifest_sha, 'provider_max_seconds': 8100,
              'execution_mode': 'API_SAVE_AND_RUN_ALL',
              'publisher_admission_status': 'QUALIFIED_AND_AUTHORIZED', 'automatic_replay': False}
    atomic_json(frozen/'SUBMISSION_INTENT.json', intent)
    return intent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', required=True, type=int)
    for name in ('staging', 'cad', 'pilot-admission', 'publisher-receipt'):
        parser.add_argument('--'+name, required=True, type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
