"""Exact source deployment through the existing authenticated CanMV client."""
import argparse,base64,hashlib,json
from pathlib import Path
from tools.deploy_native_trial import CanMvClient


def deploy(root,model_source,receipt):
    if root!='/sdcard/tfp1116':raise ValueError('Explicit production release root required')
    client=CanMvClient()
    if client.tool('canmv_script_running').get('running') is not False:raise RuntimeError('COM3 busy')
    if client.tool('canmv_stat_file',{'path':root}).get('exists') is not False:
        raise RuntimeError('Existing release preserved; reconcile before another deployment')
    project=Path(__file__).resolve().parents[1]
    files=[]
    for folder in ('k230','vision','ports','core'):
        files.extend(p for p in (project/folder).rglob('*.py') if '__pycache__' not in p.parts)
    files.append(project/'run_topface_service.py')
    names=[p.relative_to(project).as_posix() for p in files]
    dirs={root,root+'/model'}
    for name in names:
        if len((root+'/'+name).encode())>=64:raise ValueError('Native path exceeds bounded length')
        parent=Path(name).parent
        while str(parent)!='.':dirs.add(root+'/'+parent.as_posix());parent=parent.parent
    for path in sorted(dirs,key=lambda p:(p.count('/'),p)):
        if not client.tool('canmv_mkdir',{'path':path}).get('success'):raise RuntimeError('mkdir unconfirmed')
    result={'status':'INCOMPLETE','root':root,'files':[],'model_source':model_source,'model_copy':'NOT_RUN'}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    for path,name in zip(files,names):
        raw=path.read_bytes()
        if len(raw)>262144:raise ValueError('Source requires bounded chunking')
        if not client.tool('canmv_write_file',{'path':root+'/'+name,'content':base64.b64encode(raw).decode(),
                                             'encoding':'base64'}).get('success'):raise RuntimeError('write unconfirmed')
        value=client.tool('canmv_read_file',{'path':root+'/'+name,'encoding':'base64'})
        actual=base64.b64decode(value['content'],validate=True)
        if raw!=actual:raise RuntimeError('Exact source byte readback failed')
        result['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        receipt.write_text(json.dumps(result,indent=2),encoding='utf8')
    script=("import os,sys,json\nsys.path.insert(0,"+repr(root)+")\n"
            "from k230.topface_adapter import digest,MODEL_SHA\n"
            "source="+repr(model_source)+"\ntarget="+repr(root+'/model')+"\n"
            "if digest(source+'/topface.kmodel')!=MODEL_SHA:raise ValueError('Source model differs')\n"
            "for name in ('topface.kmodel','identity.json','output_contract.json'):\n"
            " with open(source+'/'+name,'rb') as src,open(target+'/'+name,'wb') as dst:\n"
            "  while True:\n   block=src.read(4096)\n   if not block:break\n   dst.write(block)\n"
            " if digest(source+'/'+name)!=digest(target+'/'+name):raise ValueError('Model copy byte mismatch')\n"
            "print('TOPFACE_PRODUCTION_COPY_VERIFIED_1116')\n")
    result['model_copy_script']=script
    client.tool('canmv_run_script',{'script':script})
    import time
    for _ in range(20):
        output=client.tool('canmv_terminal_output',{'clear':False}).get('text','')
        if 'TOPFACE_PRODUCTION_COPY_VERIFIED_1116' in output.splitlines():
            result.update(status='SOURCE_AND_MODEL_BYTES_VERIFIED',model_copy='SHA_PASS')
            receipt.write_text(json.dumps(result,indent=2),encoding='utf8');return result
        time.sleep(.2)
    raise RuntimeError('Copy completion UNKNOWN; preserve release and reconcile')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/sdcard/tfp1116')
    p.add_argument('--model-source',required=True);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();r=deploy(**vars(args));print(json.dumps({'status':r['status'],'files':len(r['files'])}))
