"""Scoped provider operations. No credential values or signed URLs are printed."""
import argparse
import json
import os
from pathlib import Path
import sys


def network():
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as k:
            enabled=winreg.QueryValueEx(k,'ProxyEnable')[0]
            if enabled:
                proxy=winreg.QueryValueEx(k,'ProxyServer')[0]
                if ';' not in proxy and '=' not in proxy:
                    if not proxy.startswith('http'):proxy='http://'+proxy
                    os.environ.update(HTTP_PROXY=proxy,HTTPS_PROXY=proxy)
    os.environ['NO_PROXY']='localhost,127.0.0.1,api.kaggle.com'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('admit','dataset','save','observe','collect','publish'))
    ap.add_argument('--record',type=Path,required=True);ap.add_argument('--stage',type=Path)
    args=ap.parse_args();network()
    root=args.record.resolve()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from cloud_provider import create_dataset,save_kernel,observe,read_output,persist_output,sha
    api=KaggleApi();api.authenticate()
    intent=json.loads((root/'SUBMISSION_INTENT.json').read_text())
    if args.action=='admit':
        from cloud_publisher import admit
        receipt=admit(intent['run_id'],intent['input_manifest_sha256'],root/'ADMISSION.json')
        intent['publisher_admission_status']='QUALIFIED_AND_AUTHORIZED'
        (root/'SUBMISSION_INTENT.json').write_text(json.dumps(intent,indent=2)+'\n')
        print(json.dumps(receipt))
    elif args.action=='dataset':create_dataset(api,args.stage.resolve(),root)
    elif args.action=='save':
        status=api.dataset_status(intent['dataset'],format='json(status,current_version_number)')
        (root/'DATASET_READY.json').write_text(json.dumps(status,indent=2,default=str)+'\n')
        save_kernel(api,root,root,False)
    else:
        saved=json.loads((root/'KERNEL_SAVE_RESPONSE.json').read_text())
        ref=saved['ref'].removeprefix('/code/');version=saved['versionNumber']
        obs=root/f'OBSERVATION-v{version}.json'
        if args.action=='observe':observe(api,ref,version,obs)
        elif args.action=='collect':
            result=json.loads(obs.read_text())
            if result['status']!='COMPLETE':raise ValueError('Exact saved version not complete')
            names=['RESULT.json','ENVIRONMENT.json',intent['run_id']+'/COMPUTE.json',intent['run_id']+'/train.log',
                   intent['run_id']+'/models/COMPARISON.json',intent['run_id']+'/models/PILOT_GATE.json']
            collected=[]
            with api.build_kaggle_client() as client:
                for name in names:
                    try:data=read_output(client,ref,version,name,8*1024**2)
                    except Exception as error:
                        collected.append({'name':name,'available':False,'error_type':type(error).__name__});continue
                    local=root/'compact'/Path(name).name;local.parent.mkdir(exist_ok=True)
                    persist_output(local,data);collected.append({'name':name,'bytes':len(data),'sha256':sha(local)})
            (root/'COLLECTION.json').write_text(json.dumps(collected,indent=2)+'\n')
            print(json.dumps(collected))
        elif args.action=='publish':
            import cloud_publisher
            from publisher.publish_completed import publish
            admission=json.loads((root/'ADMISSION.json').read_text())
            print(json.dumps(publish(root,obs,root/'compact/RESULT.json',None,admission['publisher_source_commit'])))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'RECONCILE_ORIGINAL_OPERATION','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1) from None
