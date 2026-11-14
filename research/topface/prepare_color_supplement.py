"""Prepare a finite240-scene color counterexample supplement, not a replay."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json,sha
from supplement_plan import build_plan

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main(staging,qualification=False):
    staging=Path(staging).resolve();staging.mkdir(parents=True,exist_ok=False)
    plan=build_plan()
    if qualification:plan.update(count=6,scenes=plan['scenes'][:6])
    names=['geometry.py','cloud_store.py','native_job.py','native_cad.py','render_native.py',
           'scene_config.py','supplement_plan.py','audit_v11.py','input_locator.py']
    for name in names:shutil.copy2(HERE/name,staging/name)
    shutil.copy2(ROOT/'docs/topface/v11/reports/SMOKE_HUMAN_REVIEW.json',staging/'SMOKE_HUMAN_REVIEW.json')
    shutil.copy2(Path('C:/Users/28389/Desktop/k230-project/models/k230_material_block_model_bundle.zip'),staging/'cad_bundle.bin')
    atomic_json(staging/'SCENE_PLAN.json',plan)
    run_id='k230-topface-v11-color-counterexamples-a01'
    dataset='lotus30/k230-topface-v11-colorfix-input-a01'
    kernel='lotus30/k230-topface-v11-01b-color-counterexamples'
    manifest={'schema':'topface-native-color-supplement-input-v1.1','created_utc':datetime.now(timezone.utc).isoformat(),
        'run_id':run_id,'parent_run':'k230-topface-v11-pilot-1000-a02','original1000_replayed':False,
        'new_scenes':plan['count'],'seed':20261116,'worker_max_seconds':3600,'provider_max_seconds':4200,
        'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(staging.iterdir())]}
    atomic_json(staging/'INPUT_MANIFEST.json',manifest)
    atomic_json(staging/'dataset-metadata.json',{'id':dataset,'title':'K230 Topface V11 colorfix input a01',
        'licenses':[{'name':'other'}],'description':'Private unchanged original CAD and nativebpy color-count counterexample source.240new independent scenes; original1000not replayed.Original asset rights retained.'})
    kernel_dir=staging.parent/(staging.name+'-kernel');kernel_dir.mkdir(exist_ok=False)
    name='01b_color_counterexamples.ipynb'
    intro='''# 01b color/count counterexamples — V1.1

The original1000 Pilot scenes completed, QA passed and archive was accepted on Drive. A later conditional-distribution audit found legacy colors tied to object count: one-target scenes onlyyellow, two-target scenes onlyblue+green. Preserve the original archive and scientific attempt. This job generates **240additional new scenes**, seed20261116, using the unchanged fusedCAD/palette/camera envelope. It does not repeat the original1000 or perform optimizer updates.

The immutable scene plan contains120train/60val/60synthetic_test. Each count1..5 has24train/12val/12test scenes; first-target colors1..6 appear4/2/2times per count respectively, with randomized other colors. Whole camera configurations retain the original split hash. Validation/test counterexamples remain outside training. Full-generation colors now use an independent RNG instead of index%6.

Attach existingDOASTRA_GDRIVE_OAUTH_JSON, Internet on, Save&RunAll once. Same nativeBlender4.2.9, Drive admission/one-frame qualification/QA/full-byte publication;3600s worker/4200s provider. No automatic replay. MainPilot and supplement will be joined only by a later hash-verified audit; Full training remains gated.
'''
    code=(HERE/'input_locator.py').read_text(encoding='utf8')+f'''
import sys
input_root=locate_input('/kaggle/input',{sha(staging/'INPUT_MANIFEST.json')!r})
sys.path.insert(0,str(input_root))
from native_job import main
main(input_root,'/kaggle/working',{run_id!r},count=240,seed=20261116,max_seconds=3600)
'''
    atomic_json(kernel_dir/name,{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3'}},
        'cells':[{'cell_type':'markdown','id':'colorfix-contract','metadata':{},'source':intro.splitlines(True)},
            {'cell_type':'code','id':'new-scenes-only','metadata':{},'source':code.splitlines(True),'execution_count':None,'outputs':[]}]})
    atomic_json(kernel_dir/'kernel-metadata.json',{'id':kernel,'title':'K230 Topface V11 01b Color Counterexamples',
        'code_file':name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,
        'enable_tpu':False,'enable_internet':True,'dataset_sources':[dataset+'/1'],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]})
    if not qualification:
        frozen=ROOT/'docs/topface/v11/cloud/colorfix-a01';frozen.mkdir(parents=True,exist_ok=False)
        for source in (staging/'INPUT_MANIFEST.json',staging/'SCENE_PLAN.json',kernel_dir/name,kernel_dir/'kernel-metadata.json'):
            shutil.copy2(source,frozen/source.name)
        intent={'schema':'topface-color-counterexample-intent-v1.1','status':'PREPARED_NOT_SUBMITTED',
            'run_id':run_id,'dataset':dataset,'dataset_version':1,'kernel':kernel,'notebook_file':name,
            'notebook_sha256':sha(kernel_dir/name),'input_manifest_sha256':sha(staging/'INPUT_MANIFEST.json'),
            'worker_max_seconds':3600,'provider_max_seconds':4200,'automatic_replay':False,
            'original1000_replayed':False,'new_scenes':240,'seed':20261116}
        atomic_json(frozen/'SUBMISSION_INTENT.json',intent)
    print(json.dumps({'staging':str(staging),'kernel_dir':str(kernel_dir),'count':plan['count'],'qualification':qualification}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--staging',type=Path,required=True);p.add_argument('--qualification',action='store_true')
    main(**vars(p.parse_args()))
