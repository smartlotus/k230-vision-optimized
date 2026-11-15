"""Freeze a CPU export and KModel conversion against accepted cloud inputs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import zipfile

from cloud_store import atomic_json,sha
from prepare_full_downstream import source_ref


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def prepare(phase,revision,staging,parent,runtime_receipt,runtime_qualification,
            native_runtime,dataset_acceptance=None,shard_inventory=None):
    if phase not in ('pilot_conversion','formal_conversion') or not revision.isalnum():
        raise ValueError('Explicit conversion phase and revision required')
    receipt=read(runtime_receipt)
    qualification=read(runtime_qualification)
    if (qualification['status']!='LINUX_COMPILER_FIXTURE_PASS' or
            qualification['run_id']!=receipt['run_id'] or
            qualification['archive']['sha256']!=receipt['uploaded']['sha256']):
        raise ValueError('The offline compiler has not passed its cloud qualification')
    source_ref(receipt)
    selected=read(parent)
    mounts=[source_ref(selected['publication'])]
    label='pilot' if phase=='pilot_conversion' else 'formal'
    run_id=f'k230-topface-v11-conversion-{label}-{revision}'
    dataset=f'lotus30/k230-topface-v11-conversion-{label}-input-{revision}'
    ref=f'lotus30/k230-topface-v11-07b-{label}-conversion'
    max_seconds=21600 if phase=='formal_conversion' else 3600
    config={'run_id':run_id,'phase':phase,'parent':selected,'max_seconds':max_seconds}
    if phase=='pilot_conversion':
        config['pilot_sources']={}
        for prefix,folder,name in (
            ('primary','pilot-a02/compact','k230-topface-v11-pilot-1000-a02-RESULT.json'),
            ('colorfix','colorfix-a01/compact-v2','k230-topface-v11-color-counterexamples-a01-RESULT.json'),
        ):
            result=read(ROOT/'docs/topface/v11/cloud'/folder/name)
            if result['status']!='COMPUTE_AND_QA_PASS' or result['durable_status']!='FULL_BYTE_SHA256_PASS':
                raise ValueError('Pilot data archive has not passed')
            config['pilot_sources'][prefix]={'archive':result['archive']}
        mounts+=['lotus30/k230-topface-v11-01-simulation-pilot/2',
                 'lotus30/k230-topface-v11-01b-color-counterexamples/2']
        config['dataset_index_sha256']=read(ROOT/'docs/topface/v11/cloud/audit-a02/ACCEPTANCE-v4.json')['composite_audit_sha256']
    else:
        if dataset_acceptance is None or shard_inventory is None:
            raise ValueError('Formal conversion requires the accepted Full dataset')
        audit=read(dataset_acceptance)
        if audit['status']!='FULL_DATASET_ACCEPTED' or sha(shard_inventory)!=audit['shard_inventory_sha256']:
            raise ValueError('Full dataset inventory differs from the selected checkpoint')
        inventory=read(shard_inventory)
        config.update(shards=inventory['shards'],total_scenes=inventory['total_planned'],
                      data_seed=inventory['seed'],dataset_index_sha256=audit['dataset_index_sha256'])
        mounts.extend(source_ref(row['publication']) for row in inventory['shards'])
        if selected.get('real_background_bank') is not None:
            config['ptq_background_bank']=selected['real_background_bank']
            config['training_input']=selected['training_input']
            # The selected formal output retains an exact source copy, including
            # this INPUT_MANIFEST and background bank. Reuse that mounted copy;
            # mounting the original dataset too would make the hash locator ambiguous.
    staging=staging.resolve()
    frozen=ROOT/f'docs/topface/v11/cloud/conversion-{label}-{revision}'
    kernel=staging.with_name(staging.name+'-kernel')
    if any(path.exists() for path in (staging,frozen,kernel)):
        raise FileExistsError('Preserve the existing conversion preparation')
    staging.mkdir(parents=True);frozen.mkdir(parents=True);kernel.mkdir()
    names=('compiler_runtime.py','native_export_runtime.py','model_artifact_job.py','cloud_artifacts.py','cloud_store.py',
           'downstream_job.py','formal_job.py','full_dataset.py','native_job.py','combine_datasets.py',
           'scene_config.py','geometry.py','dataset_diagnostics.py','task_geometry.py',
           'nanodet_topface.py','export_nanodet_candidate.py','export_trained_topface.py',
           'compile_trained_topface.py','evaluate_task.py','evaluate_final.py','task_training.py','input_locator.py',
           'nncase_simulator.py','simulate_dataset.py','evaluate_saved_raw.py','background_augmentation.py')
    for name in names:
        shutil.copy2(HERE/name,staging/name)
    native=read(native_runtime/'NATIVE_EXPORT_RUNTIME.json')
    for row in native['artifacts']:
        path=native_runtime/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Native export runtime differs from its frozen wheel manifest')
        shutil.copy2(path,staging/path.name)
    shutil.copy2(native_runtime/'NATIVE_EXPORT_RUNTIME.json',staging/'NATIVE_EXPORT_RUNTIME.json')
    for name in ('package_trial.py','topface_trial.py','topface_decode.py'):
        shutil.copy2(HERE/'board'/name,staging/name)
    with zipfile.ZipFile(staging/'nanodet-source.bin','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((HERE/'vendor/nanodet_core').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                archive.write(path,path.relative_to(HERE).as_posix())
    atomic_json(staging/'MODEL_ARTIFACT_CONFIG.json',config)
    manifest={'schema':'topface-model-artifact-input-v1.1','run_id':run_id,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'files':[{'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}
                       for path in sorted(staging.iterdir())]}
    atomic_json(staging/'INPUT_MANIFEST.json',manifest)
    atomic_json(staging/'dataset-metadata.json',{'id':dataset,'title':f'K230 Topface V11 {label} conversion input',
        'licenses':[{'name':'other'}],'description':'Private byte-pinned model conversion source and cloud artifact identities.'})
    runtime_input=read(ROOT/'docs/topface/v11/cloud/compiler-linux-a01/SUBMISSION_INTENT.json')
    training_root=(f"locate_input('/kaggle/input',{config['training_input']['manifest_sha256']!r})"
                   if 'training_input' in config else 'None')
    code=(HERE/'input_locator.py').read_text(encoding='utf8')+f'''
import sys
input_root=locate_input('/kaggle/input',{sha(staging/'INPUT_MANIFEST.json')!r})
runtime_root=locate_input('/kaggle/input',{runtime_input['input_manifest_sha256']!r})
sys.path.insert(0,str(input_root))
from model_artifact_job import run
run(input_root,runtime_root,{run_id!r},training_input_root={training_root})
'''
    name='07b_model_conversion.ipynb'
    atomic_json(kernel/name,{'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
        'cells':[{'cell_type':'code','id':'model-build','metadata':{},'source':code,'execution_count':None,'outputs':[]}]})
    atomic_json(kernel/'kernel-metadata.json',{'id':ref,'title':f'K230 Topface V11 07b {label.title()} Conversion',
        'code_file':name,'language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':False,'enable_internet':False,'dataset_sources':[dataset+'/1',runtime_input['dataset']+'/1'],
        'kernel_sources':mounts,'competition_sources':[]})
    for source,files in ((staging,('MODEL_ARTIFACT_CONFIG.json','INPUT_MANIFEST.json')),(kernel,(name,'kernel-metadata.json'))):
        for file in files:
            shutil.copy2(source/file,frozen/file)
    intent={'run_id':run_id,'kernel':ref,'dataset':dataset,'dataset_version':1,'notebook_file':name,
        'notebook_sha256':sha(kernel/name),'input_manifest_sha256':sha(staging/'INPUT_MANIFEST.json'),
        'provider_max_seconds':max_seconds+900,'execution_mode':'API_SAVE_AND_RUN_ALL',
        'publisher_admission_status':'QUALIFIED_AND_AUTHORIZED','automatic_replay':False}
    atomic_json(frozen/'SUBMISSION_INTENT.json',intent)
    return intent


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('pilot_conversion','formal_conversion'),required=True)
    parser.add_argument('--revision',required=True)
    for key in ('staging','parent','runtime-receipt','runtime-qualification','native-runtime'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('dataset-acceptance','shard-inventory'):
        parser.add_argument('--'+key,type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())),indent=2))
