"""Bounded exact-version checkpoint recovery and official offline ORT wheels."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from cloud import network,authenticate
from cloud_provider import read_output, persist_output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();network()
    args.out.mkdir(parents=True,exist_ok=False)
    intent=json.loads((args.record/'SUBMISSION_INTENT.json').read_text())
    saved=json.loads((args.record/'KERNEL_SAVE_RESPONSE.json').read_text())
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();authenticate(api);inventory=[]
    with api.build_kaggle_client() as client:
        raw=read_output(client,saved['ref'].removeprefix('/code/'),saved['versionNumber'],intent['run_id']+'/MEMBERS.json',1024**2)
        members={r['name']:r for r in json.loads(raw)}
        for name in ('best.pt','last.pt','history.json','camera_xy.onnx'):
            remote='models/cnn1/'+name;row=members[remote]
            data=read_output(client,saved['ref'].removeprefix('/code/'),saved['versionNumber'],intent['run_id']+'/'+remote,10*1024**2)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('Original member digest mismatch')
            target=args.out/'prior_cnn1'/name;target.parent.mkdir(exist_ok=True)
            persist_output(target,data);inventory.append(row)
    (args.out/'RECOVERED.json').write_text(json.dumps({'source':saved,'files':inventory,'automatic_replay':False},indent=2)+'\n')
    deps=args.out/'deps';deps.mkdir()
    with urllib.request.urlopen('https://pypi.org/pypi/onnxruntime/1.24.4/json',timeout=30) as f:metadata=json.load(f)
    wheels=[]
    for abi in ('cp311','cp312','cp313'):
        row=next(x for x in metadata['urls'] if f'-{abi}-{abi}-manylinux' in x['filename'] and '_x86_64' in x['filename'])
        target=deps/row['filename']
        with urllib.request.urlopen(row['url'],timeout=60) as f:payload=f.read(25*1024**2)
        if len(payload)!=row['size'] or hashlib.sha256(payload).hexdigest()!=row['digests']['sha256']:
            raise ValueError('Official wheel digest mismatch')
        target.write_bytes(payload)
        wheels.append({'name':row['filename'],'bytes':len(payload),'sha256':row['digests']['sha256'],'source':'official PyPI onnxruntime1.24.4'})
    (deps/'WHEELS.json').write_text(json.dumps(wheels,indent=2)+'\n')
    print(json.dumps({'recovered_members':len(inventory),'official_wheels':len(wheels),'source_version':saved['versionNumber']}))


if __name__=='__main__':main()
