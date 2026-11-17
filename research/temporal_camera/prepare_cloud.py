"""Freeze byte-pinned private Pilot input and fixed Kaggle execution source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from dataset import audit, digest


def put(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--attempt',default='a01');ap.add_argument('--recovery',type=Path)
    ap.add_argument('--encoder',choices=('baseline','spatial'),default='baseline')
    ap.add_argument('--learning',choices=('ce','xy_smooth'),default='ce')
    ap.add_argument('--settings',default='cnn1,gru4,gru8,tcn4,tcn8')
    ap.add_argument('--recipe',choices=('tiny','map_reuse'),default='tiny')
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--parent',type=Path)
    args=ap.parse_args();here=Path(__file__).resolve().parent
    stage=args.stage.resolve();stage.mkdir(parents=True,exist_ok=False)
    record=args.record.resolve();record.mkdir(parents=True,exist_ok=False)
    root=stage/'assembly';root.mkdir();shutil.copytree(args.data,root/'data')
    rows=[json.loads(x) for x in (root/'data/manifest.jsonl').read_text().splitlines()];audit(rows,root/'data')
    for name in ('models.py','filter.py','dataset.py','train.py','cloud_worker.py'):
        shutil.copy2(here/name,root/name)
    if args.encoder=='spatial':shutil.copy2(here/'models_spatial.py',root/'models_spatial.py')
    if args.learning=='xy_smooth':shutil.copy2(here/'learning.py',root/'learning.py')
    if args.recipe=='map_reuse':
        shutil.copy2(here/'map_models.py',root/'map_models.py')
        shutil.copy2(here/'embedded_filter.py',root/'embedded_filter.py')
        shutil.copy2(here/'train_map_temporal.py',root/'train.py')
        shutil.copy2(args.parent or here/'reused/map-inference.pt',root/'parent.pt')
        shutil.copy2(here/'reused/SOURCE.json',root/'PARENT_SOURCE.json')
        shutil.copy2(here/'competition_paths.py',root/'competition_paths.py')
        if args.parent:
            shutil.copy2(args.parent.with_name('REFERENCE.json'),root/'REFERENCE.json')
    shutil.copy2(here.parent/'topface/cloud_store.py',root/'cloud_store.py')
    if args.recovery:
        for name in (('prior_cnn1','deps') if args.encoder=='baseline' else ('deps',)):
            shutil.copytree(args.recovery/name,root/name)
        shutil.copy2(args.recovery/'RECOVERED.json',root/'RECOVERED.json')
    manifest=[{'name':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
              for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    put(root/'INPUT_MANIFEST.json',manifest);input_sha=digest(root/'INPUT_MANIFEST.json')
    expanded=sum(item['bytes'] for item in manifest)+(root/'INPUT_MANIFEST.json').stat().st_size
    payload=stage/'payload.bin'
    with tarfile.open(payload,'w:gz') as tar:
        for item in manifest:tar.add(root/item['name'],arcname=item['name'],recursive=False)
        tar.add(root/'INPUT_MANIFEST.json',arcname='INPUT_MANIFEST.json')
    payload_sha=digest(payload)
    # Only the opaque payload and metadata enter Kaggle input; assembly remains local.
    upload=stage/'upload';upload.mkdir();shutil.copy2(payload,upload/'payload.bin')
    ref='lotus30/k230-camera-xy-pilot-v1-20261117-'+args.attempt
    kernel='lotus30/k230-camera-xy-multiroute-pilot-'+args.attempt
    run_id='k230-camera-xy-multiroute-pilot-20261117-'+args.attempt
    put(upload/'dataset-metadata.json',{'id':ref,'title':'K230 Camera XY Pilot V1 20261117 '+args.attempt.upper(),
        'licenses':[{'name':'CC0-1.0'}]})
    put(upload/'CAPSULE.json',{'payload_bytes':payload.stat().st_size,'payload_sha256':payload_sha,'manifest_sha256':input_sha})
    wall=2200 if args.encoder=='spatial' else 2350 if args.recovery else 2400
    code=f'''import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
os.chdir('/kaggle/working')
files=list(Path('/kaggle/input').rglob('payload.bin'))
assert len(files)==1
p=files[0]
assert p.stat().st_size=={payload.stat().st_size} and hashlib.sha256(p.read_bytes()).hexdigest()=={payload_sha!r}
root=Path('/kaggle/working/input');root.mkdir(exist_ok=False)
with tarfile.open(p,'r:gz') as tar:
    members=tar.getmembers()
    assert len(members)=={len(manifest)+1} and sum(m.size for m in members)=={expanded}
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    tar.extractall(root,filter='data')
manifest=root/'INPUT_MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=={input_sha!r}
for item in json.loads(manifest.read_text()):
    path=root/item['name']
    assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
deps=root/'deps'
if deps.exists():
    abi='cp'+str(sys.version_info.major)+str(sys.version_info.minor)
    matches=list(deps.glob('*-'+abi+'-'+abi+'-*.whl'));assert len(matches)==1
    site=Path('/kaggle/working/ortsite')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(site),str(matches[0])],check=True)
    sys.path.insert(0,str(site));os.environ['PYTHONPATH']=str(site)
import numpy as np,onnxruntime as ort,onnx,torch
probe=onnx.helper.make_model(onnx.helper.make_graph([onnx.helper.make_node('Identity',['x'],['y'])],'ort_probe',
    [onnx.helper.make_tensor_value_info('x',onnx.TensorProto.FLOAT,[1])],[onnx.helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[1])]),opset_imports=[onnx.helper.make_opsetid('',13)])
probe.ir_version=8
assert ort.InferenceSession(probe.SerializeToString(),providers=['CPUExecutionProvider']).run(None,{{'x':np.array([1.],np.float32)}})[0][0]==1
assert torch.cuda.is_available() and torch.ones(1,device='cuda').item()==1
Path('/kaggle/working/ENVIRONMENT.json').write_text(json.dumps({{'python':sys.version,'torch':torch.__version__,'onnxruntime':ort.__version__,'gpu':torch.cuda.get_device_name(0),'actual_ort_operator_probe':'PASS'}}))
command=[sys.executable,str(root/'cloud_worker.py'),'--root',str(root),'--run-id',{run_id!r},'--input-sha',{input_sha!r},'--wall-seconds',{str(wall)!r},'--encoder',{args.encoder!r}]
command.extend(['--learning',{args.learning!r}])
command.extend(['--settings',{args.settings!r}])
command.extend(['--epochs',{str(args.epochs)!r}])
if {args.encoder=='spatial'!r}:command.append('--all-endpoints')
subprocess.run(command,check=True)
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'code','metadata':{},'source':code,'execution_count':None,'outputs':[]}]}
    put(record/'run.ipynb',notebook)
    put(record/'kernel-metadata.json',{'id':kernel,'title':'K230 Camera XY Multiroute Pilot '+args.attempt.upper(),'code_file':'run.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
        'dataset_sources':[ref+'/1'],'kernel_sources':[],'competition_sources':[]})
    put(record/'SUBMISSION_INTENT.json',{'run_id':run_id,'kernel':kernel,'dataset':ref,'dataset_version':1,
        'notebook_file':'run.ipynb','notebook_sha256':digest(record/'run.ipynb'),'input_manifest_sha256':input_sha,
        'provider_max_seconds':3000,'publisher_admission_status':'PENDING','publication_mode':'INDEPENDENT_CLOUD_PUBLISHER',
        'automatic_retry':False,'scientific_budget':{'settings':args.settings.split(','),
           'epochs_max_each':args.epochs,'early_stop_patience':12,'wall_seconds':wall,'seed':117},
        'recovery':bool(args.recovery) and args.encoder=='baseline',
        'recovery_scope':'completed CNN plus dependencies' if args.recovery and args.encoder=='baseline' else 'dependency wheels only' if args.recovery else 'none',
        'prior_cnn_optimizer_updates':96 if args.recovery and args.encoder=='baseline' else 0,
        'encoder':args.encoder,'all_endpoints':args.encoder=='spatial',
        'learning':args.learning,
        'recipe':args.recipe,
        'parent_sha256':digest(root/'parent.pt') if args.recipe=='map_reuse' else None,
        'full_generation':'NOT RUN; requires pilot acceptance'})
    shutil.copy2(root/'INPUT_MANIFEST.json',record/'INPUT_MANIFEST.json')
    print(json.dumps({'run_id':run_id,'upload':str(upload),'input_sha256':input_sha,'payload_bytes':payload.stat().st_size}))


if __name__=='__main__':main()
