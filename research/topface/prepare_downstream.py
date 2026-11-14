"""Create immutable 02 audit / 03 training inputs only after parent acceptance.

Preparation has no cloud side effect. Missing actual receipts fail closed;
placeholders cannot accidentally launch optimizer work.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json,sha
from nanodet_topface import CHECKPOINT_SHA256

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main(phase,revision,pilot_result,staging,audit_acceptance=None,supplement_result=None):
    if phase not in ('audit','training') or not revision.isalnum():raise ValueError('Explicit phase/revision required')
    parent=json.loads(Path(pilot_result).read_text(encoding='utf8'))
    if parent['status']!='COMPUTE_AND_QA_PASS' or parent['images_rendered']!=1000 or parent['durable_status']!='FULL_BYTE_SHA256_PASS':
        raise ValueError('Cannot prepare downstream execution without actual accepted Pilot')
    if parent['drive_archive']['status']!='FULL_BYTE_SHA256_PASS':raise ValueError('Parent full-byte archive acceptance missing')
    source_receipt=json.loads((ROOT/'docs/topface/v11/cloud/pilot-a02/EXECUTED_SOURCE-v2.json').read_text(encoding='utf8'))
    if source_receipt['status']!='EXECUTED_VERSION_SOURCE_MATCH' or parent['run_id']!=source_receipt['run_id']:
        raise ValueError('Parent run differs from verified exact Kaggle source')
    staging=Path(staging).resolve();staging.mkdir(parents=True,exist_ok=False)
    ordinal='02' if phase=='audit' else '03'
    run_id=f'k230-topface-v11-{phase}-{revision}'
    dataset=f'lotus30/k230-topface-v11-{phase}-input-{revision}'
    kernel=f'lotus30/k230-topface-v11-{ordinal}-'+('dataset-audit' if phase=='audit' else 'pilot-training')
    folder='1MwltNmaI24HjvcxnG1w608qui42RAPKf'
    config={'schema':'topface-downstream-config-v1.1','phase':phase,'run_id':run_id,
        'folder_id':folder,'pilot_run_id':parent['run_id'],'pilot_compute':parent,
        'pilot_provider':{'ref':'lotus30/k230-topface-v11-01-simulation-pilot','id':145002634,'version':2},
        'max_seconds':1800 if phase=='audit' else 9000,'provider_max_seconds':2400 if phase=='audit' else 10800,
        'worker_budget':'includes source fetch, diagnostics, finite experiment(s), archive, full Drive readback',
        'automatic_replay':False,'full_admission':False}
    if supplement_result:
        supplement=json.loads(Path(supplement_result).read_text(encoding='utf8'))
        if supplement['status']!='COMPUTE_AND_QA_PASS' or supplement['images_rendered']!=240 or supplement['durable_status']!='FULL_BYTE_SHA256_PASS':
            raise ValueError('Supplement compute/QA/storage has not passed')
        config['supplement_compute']=supplement
    files=['geometry.py','scene_config.py','cloud_store.py','native_job.py','cloud_artifacts.py','combine_datasets.py',
           'downstream_job.py','dataset_diagnostics.py','input_locator.py']
    if phase=='training':
        if audit_acceptance is None:raise ValueError('Independent02 audit receipt required before training preparation')
        acceptance=json.loads(Path(audit_acceptance).read_text(encoding='utf8'))
        if acceptance['training_admission']!='PASS' or acceptance['pilot_archive_sha256']!=parent['drive_archive']['sha256']:
            raise ValueError('Independent02 audit did not accept the same parent archive')
        config['audit_acceptance']=acceptance
        files+=['task_geometry.py','task_training.py','evaluate_task.py','train_experiment.py','nanodet_topface.py']
    for name in files:shutil.copy2(HERE/name,staging/name)
    if phase=='training':
        shutil.copytree(HERE/'vendor/nanodet_core',staging/'vendor/nanodet_core',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        checkpoint=ROOT/'.ops/local/topface-g3-models/nanodet-plus-m_320_checkpoint.ckpt'
        if sha(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Pretrained checkpoint identity changed')
        shutil.copy2(checkpoint,staging/'nanodet-pretrained.bin')
    atomic_json(staging/'DOWNSTREAM_CONFIG.json',config)
    members=[p for p in sorted(staging.rglob('*')) if p.is_file()]
    manifest={'schema':'topface-downstream-input-v1.1','phase':phase,'run_id':run_id,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'files':[{'path':p.relative_to(staging).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in members]}
    atomic_json(staging/'INPUT_MANIFEST.json',manifest)
    atomic_json(staging/'dataset-metadata.json',{'id':dataset,'title':f'K230 TopFace V11 {phase} input {revision}',
        'licenses':[{'name':'other'}],'description':'Private byte-pinned K230 research input. Upstream NanoDet modules retain Apache-2.0 and original authors notices. Original assets are not publicly redistributed.'})
    kernel_dir=staging.parent/(staging.name+'-kernel');kernel_dir.mkdir(exist_ok=False)
    code=(HERE/'input_locator.py').read_text(encoding='utf8')+f'''
import sys
input_root=locate_input('/kaggle/input',{sha(staging/'INPUT_MANIFEST.json')!r})
sys.path.insert(0,str(input_root))
from downstream_job import main
main(input_root,{run_id!r},{phase!r},max_seconds={config['max_seconds']})
'''
    intro=f'''# {ordinal}_{phase} — V1.1

Private input version1, frozen parent Pilot Kaggle145002634 v2, run `{parent['run_id']}`, Drive archive SHA `{parent['drive_archive']['sha256']}`. The cloud worker reads that immutable archive directly from Drive, verifies all bytes and archive members, then runs this phase. Raw data and checkpoints stay in cloud; host collects bounded receipts only.

Attach existing `DOASTRA_GDRIVE_OAUTH_JSON`, enable Internet, and run once. A persistent Drive admission marker prevents accidental repeated science in a new session. Finite worker budget: {config['max_seconds']}s including publication. No Full generation or formal training is admitted by this notebook.

''' + ('''Audit uses no GPU and performs no optimizer update. It checks class/scale/visibility/negative/near-duplicate/group leakage and preserves all0–5 world-instance denominators. Perceptual similarities are review candidates, not automatic deletion.
''' if phase=='audit' else '''Training uses CUDA0, FP32, no dependency installs and no large hue shifts. Logical sequence: 16 train images overfit (max1000updates), fresh pretrained20epoch Pilot, fresh random-initialization20epoch controlled comparison. Overfit failure stops before Pilot training. The two Pilot runs use identical splits, augmentation seeds, hyperparameters and epoch caps. Only val selects best; no synthetic_test or real test evaluation. Save best/last/optimizer/RNG/TensorBoard/config/metrics/SHA; failed computations are archived and never automatically replayed.
''')
    name=f'{ordinal}_{phase}.ipynb'
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','id':'contract','metadata':{},'source':intro.splitlines(True)},
            {'cell_type':'code','id':'finite-worker','metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]}]}
    atomic_json(kernel_dir/name,notebook)
    title='K230 Topface V11 '+ordinal+(' Dataset Audit' if phase=='audit' else ' Pilot Training')
    atomic_json(kernel_dir/'kernel-metadata.json',{'id':kernel,'title':title,'code_file':name,
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':phase=='training',
        'enable_tpu':False,'enable_internet':True,'dataset_sources':[dataset+'/1'],
        'kernel_sources':[],'competition_sources':[],'model_sources':[]})
    frozen=ROOT/f'docs/topface/v11/cloud/{phase}-{revision}';frozen.mkdir(parents=True,exist_ok=False)
    for file in ('INPUT_MANIFEST.json','DOWNSTREAM_CONFIG.json'):shutil.copy2(staging/file,frozen/file)
    for file in (name,'kernel-metadata.json'):shutil.copy2(kernel_dir/file,frozen/file)
    intent={'schema':'topface-downstream-save-intent-v1.1','status':'PREPARED_NOT_SUBMITTED','phase':phase,
        'run_id':run_id,'dataset':dataset,'dataset_version':1,'kernel':kernel,
        'notebook_file':name,'notebook_sha256':sha(kernel_dir/name),
        'input_manifest_sha256':sha(staging/'INPUT_MANIFEST.json'),
        'provider_max_seconds':config['provider_max_seconds'],'automatic_replay':False,
        'parent_pilot_archive_sha256':parent['drive_archive']['sha256']}
    atomic_json(frozen/'SUBMISSION_INTENT.json',intent)
    print(json.dumps({'staging':str(staging),'kernel_dir':str(kernel_dir),'frozen':str(frozen),**intent},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','training'],required=True)
    p.add_argument('--revision',required=True);p.add_argument('--pilot-result',type=Path,required=True)
    p.add_argument('--staging',type=Path,required=True);p.add_argument('--audit-acceptance',type=Path)
    p.add_argument('--supplement-result',type=Path)
    main(**vars(p.parse_args()))
