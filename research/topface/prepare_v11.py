"""Stage the approved V1.1 native Pilot. Does not submit or execute Kaggle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json, sha

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'docs/topface/v11'
DATASET='lotus30/k230-topface-native-v11-input-20261114'
KERNEL='lotus30/k230-topface-v11-01-simulation-pilot'
RUN_ID='k230-topface-v11-pilot-1000-a01'


def main(staging, revision='a02', qualification=False):
    if not revision.isalnum():
        raise ValueError('Revision must be alphanumeric')
    dataset=DATASET+'-'+revision
    run_id='k230-topface-v11-pilot-1000-'+revision
    staging=Path(staging).resolve()
    if staging.exists():
        raise ValueError('Use a new immutable staging directory')
    staging.mkdir(parents=True)
    files=['geometry.py','scene_config.py','native_cad.py','render_native.py',
           'audit_v11.py','native_job.py','cloud_store.py','input_locator.py']
    for name in files:
        shutil.copy2(HERE/name,staging/name)
    shutil.copy2(DOC/'reports/SMOKE_HUMAN_REVIEW.json',staging/'SMOKE_HUMAN_REVIEW.json')
    shutil.copy2(DOC/'camera/camera_profile.yaml',staging/'camera_profile.yaml')
    shutil.copy2(DOC/'camera/camera_sweep.yaml',staging/'camera_sweep.yaml')
    cad=Path('C:/Users/28389/Desktop/k230-project/models/k230_material_block_model_bundle.zip')
    shutil.copy2(cad,staging/'cad_bundle.bin')
    manifest={'schema':'topface-native-v1.1-input','created_utc':datetime.now(timezone.utc).isoformat(),
        'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(staging.iterdir())],
        'cad_original_sha256':sha(cad),'pilot_run_id':run_id,'count':1000,'seed':20261115,
        'max_runtime_seconds':9000,'provider_timeout_seconds':10800,
        'storage_route':'runtime-attached existing Secret -> Drive with full-byte readback',
        'input_migration':'one-time private source/CAD input; newly rendered artifacts stay in cloud',
        'formal_training_started':False,'legacy_v8_restart':False}
    atomic_json(staging/'INPUT_MANIFEST.json',manifest)
    atomic_json(staging/'dataset-metadata.json',{'id':dataset,
        'title':'K230 TopFaceNet native V1.1 input 20261114','licenses':[{'name':'other'}],
        'description':'Private original CAD and independently written native bpy Pilot source. Original asset rights retained. No open redistribution licence is granted. Native Blender only; no BlenderProc install.'})
    frozen=DOC/('cloud/pilot-'+revision)
    if not qualification:
        frozen.mkdir(parents=True,exist_ok=False)
        atomic_json(frozen/'INPUT_MANIFEST.json',manifest)
    code=(HERE/'input_locator.py').read_text(encoding='utf8')+f'''
import sys
input_root=locate_input('/kaggle/input',{sha(staging/'INPUT_MANIFEST.json')!r})
sys.path.insert(0,str(input_root))
from native_job import main
main(input_root,'/kaggle/working',{run_id!r},count=1000,seed=20261115,max_seconds=9000)
'''
    intro='''# 01_topface_simulation — V1.1 approved Pilot

100 independent local Smoke scenes passed automatic QA and the user reviewed 30 previews and authorized Pilot. This version renders **1,000 new independently seeded scenes**, 0–5 upright blocks, six colors, lens-to-top 100–400 mm, camera angle 80–100 degrees to the top plane. Actual intrinsics/focus remain uncalibrated.

Attach the existing **DOASTRA_GDRIVE_OAUTH_JSON** Secret, enable Internet, and Save & Run All once. The job checks Drive write/readback before downloading the SHA-pinned Blender 4.2.9 Linux runtime. It has no BlenderProc install or quickstart. PyTorch GPU and actual Cycles GPU are reported separately. A one-frame runtime qualification precedes the 1,000-scene run, with at most one CPU fallback for that qualification. A failed or partial Pilot is archived and is never replayed automatically.

Budget: 9,000 s worker / 10,800 s provider session; at least 8 GiB free working space; 24 samples; seed 20261115. Complete RGB, top masks, physical projected centers, conic ellipse diagnostics, labels, overlays, source and member SHA manifest are archived from Kaggle directly to the K230 Drive folder and fully read back. No optimizer updates occur here. Do not run the retired v8 notebook.
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},
        'cells':[{'cell_type':'markdown','id':'v11-contract','metadata':{},'source':intro.splitlines(True)},
                 {'cell_type':'code','id':'finite-pilot','metadata':{},'source':code.splitlines(True),
                  'execution_count':None,'outputs':[]}]}
    kernel=staging.parent/(staging.name+'-kernel')
    kernel.mkdir(exist_ok=False)
    atomic_json(kernel/'01_topface_simulation.ipynb',notebook)
    atomic_json(kernel/'kernel-metadata.json',{'id':KERNEL,
        'title':'K230 Topface V11 01 Simulation Pilot',
        'code_file':'01_topface_simulation.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
        'dataset_sources':[dataset+'/1'],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
    if not qualification:
        shutil.copy2(kernel/'01_topface_simulation.ipynb',frozen/'01_topface_simulation.ipynb')
        shutil.copy2(kernel/'kernel-metadata.json',frozen/'kernel-metadata.json')
    intent={'schema':'topface-submission-intent-v1.1','status':'PREPARED_NOT_SUBMITTED',
        'dataset':dataset,'required_dataset_version':1,'kernel':KERNEL,'run_id':run_id,
        'input_manifest_sha256':sha(staging/'INPUT_MANIFEST.json'),
        'notebook_sha256':sha(kernel/'01_topface_simulation.ipynb'),
        'provider_mode':'QUICK_SAVE first; editor SAVE_AND_RUN_ALL after runtime Secret association',
        'scientific_runs_admitted':0,'count':1000,'seed':20261115,
        'worker_max_seconds':9000,'provider_max_seconds':10800,'automatic_replay':False,
        'secret_name':'DOASTRA_GDRIVE_OAUTH_JSON',
        'folder_id':'1MwltNmaI24HjvcxnG1w608qui42RAPKf',
        'storage_admission':'required inside exact runtime before download/render',
        'old_kernel':'retired; no restart authorized by this new submission'}
    if not qualification:
        atomic_json(frozen/'SUBMISSION_INTENT.json',intent)
    print(json.dumps({'staging':str(staging),'kernel':str(kernel),**intent},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--staging',type=Path,required=True)
    parser.add_argument('--revision',default='a02');parser.add_argument('--qualification',action='store_true')
    args=parser.parse_args();main(args.staging,args.revision,args.qualification)
