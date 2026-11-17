"""Collect named bounded deployment models, verified against saved member hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from cloud import network,authenticate
from cloud_provider import read_output,persist_output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--settings',default='cnn1,gru4,gru8,tcn4,tcn8')
    args=ap.parse_args();network()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();authenticate(api)
    record=args.record.resolve();intent=json.loads((record/'SUBMISSION_INTENT.json').read_text())
    saved=json.loads((record/'KERNEL_SAVE_RESPONSE.json').read_text());ref=saved['ref'].removeprefix('/code/')
    observation=json.loads((record/f"OBSERVATION-v{saved['versionNumber']}.json").read_text())
    if observation['status']!='COMPLETE' or observation['kernel_id']!=saved['kernelId']:
        raise ValueError('Original exact version not complete')
    args.out.mkdir(parents=True,exist_ok=True);receipt=[];total=0
    with api.build_kaggle_client() as client:
        raw=read_output(client,ref,saved['versionNumber'],intent['run_id']+'/MEMBERS.json',2*1024**2)
        members={m['name']:m for m in json.loads(raw)}
        for setting in args.settings.split(','):
            if setting not in ('cnn1','gru4','gru8','tcn4','tcn8'):raise ValueError('Invalid named setting')
            for name in ('best.pt','camera_xy.onnx','METRICS.json','history.json'):
                remote=f'models/{setting}/{name}';m=members[remote]
                if m['bytes']>12*1024**2:raise ValueError('Model member exceeds bound')
                total+=m['bytes']
                if total>100*1024**2:raise ValueError('Named deployment collection exceeds100MiB')
                target=args.out/setting/name;target.parent.mkdir(exist_ok=True)
                if target.exists():data=target.read_bytes()
                else:data=read_output(client,ref,saved['versionNumber'],intent['run_id']+'/'+remote,12*1024**2)
                if len(data)!=m['bytes'] or hashlib.sha256(data).hexdigest()!=m['sha256']:
                    raise ValueError('Saved deployment member digest differs')
                persist_output(target,data);receipt.append({'setting':setting,**m})
    value={'status':'ORIGINAL_NAMED_MODEL_MEMBERS_SHA256_PASS','source':saved,'run_id':intent['run_id'],
           'total_bytes':total,'files':receipt,'full_archive_downloaded_to_host':False}
    (args.out/'MODEL_COLLECTION.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'status':value['status'],'files':len(receipt),'bytes':total}))


if __name__=='__main__':main()
