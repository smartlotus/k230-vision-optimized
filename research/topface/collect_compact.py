"""Collect bounded reports from a specific Kaggle version."""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path

from cloud_store import atomic_json


def download(client,ref,version,name,cap):
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    owner,slug=ref.split('/')
    request=ApiDownloadKernelOutputRequest();request.owner_slug=owner;request.kernel_slug=slug
    request.version_number=version;request.file_path=name
    with client.kernels.kernels_api_client.download_kernel_output(request) as response:
        response.raise_for_status()
        if int(response.headers.get('Content-Length','0'))>cap:
            raise ValueError('Saved report exceeds local byte cap')
        data=bytearray()
        for chunk in response.iter_content(16384):
            data.extend(chunk)
            if len(data)>cap:raise ValueError('Saved report stream exceeds local byte cap')
    return bytes(data)


def collect(ref,version,run_id,output,token_file,publisher_route=False,similarity=False,training=False):
    if not run_id.replace('-','').isalnum() or version<1:raise ValueError('Explicit original run/version required')
    os.environ['KAGGLE_API_TOKEN']=Path(token_file).read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with api.build_kaggle_client() as client:
        for suffix in (('RESULT',) if publisher_route else ('RESULT','DRIVE')):
            name=run_id+'-'+suffix+'.json';target=output/name
            if target.exists():
                value=json.loads(target.read_text(encoding='utf8'))
                if value.get('run_id')!=run_id:raise ValueError('Existing compact identity mismatch')
                rows.append({'name':name,'status':'ALREADY_COLLECTED','sha256':hashlib.sha256(target.read_bytes()).hexdigest()});continue
            try:
                data=download(client,ref,version,name,256*1024)
                value=json.loads(data)
                if value.get('run_id')!=run_id:raise ValueError('Downloaded compact run identity mismatch')
                temporary=target.with_suffix('.json.part');temporary.write_bytes(data);temporary.replace(target)
                rows.append({'name':name,'status':'COLLECTED','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            except Exception as error:
                # Source not saved yet is an observation failure, never evidence
                # to rerun science. No signed URL/exception strings in receipt.
                rows.append({'name':name,'status':'NOT_AVAILABLE_OR_FETCH_FAILED','error_type':type(error).__name__})
        if similarity:
            result=json.loads((output/(run_id+'-RESULT.json')).read_text(encoding='utf8'))
            expected=result['similarity_review_evidence']
            path=(f'{run_id}/similarity-review/' if result.get('schema')=='topface-full-audit-v1.1'
                  else f'{run_id}/result/similarity-review/')
            name='NEAR_SIMILARITY_EVIDENCE.json'
            raw=download(client,ref,version,path+name,256*1024)
            report=json.loads(raw)
            if len(report['pairs'])!=expected['pairs'] or report['previews']!=expected['previews']:
                raise ValueError('Similarity report does not match the compact result')
            if len(report['previews'])>4:raise ValueError('Unexpected number of similarity previews')
            for number,row in enumerate(report['previews'],1):
                image_name=f'near-similarity-{number:02d}.jpg'
                if row['name']!=image_name:raise ValueError('Invalid similarity preview name')
                picture=download(client,ref,version,path+image_name,1024**2)
                if len(picture)!=row['bytes'] or hashlib.sha256(picture).hexdigest()!=row['sha256']:
                    raise ValueError('Similarity preview differs from compact result')
                temporary=output/(image_name+'.part')
                temporary.write_bytes(picture);temporary.replace(output/image_name)
                rows.append({'name':image_name,'status':'COLLECTED','bytes':len(picture),'sha256':row['sha256']})
            atomic_json(output/name,report)
        if training:
            result=json.loads((output/(run_id+'-RESULT.json')).read_text(encoding='utf8'))
            allowed=('overfit','pilot_pretrained','pilot_random_control','pilot_geometry_refine')
            for experiment in result.get('experiments',[]):
                kind=experiment['kind']
                if kind not in allowed:raise ValueError('Unexpected training report')
                if experiment['best_epoch'] is None:continue
                prefix=f'{run_id}/result/{kind}/'
                data=download(client,ref,version,prefix+'BEST_METRICS.json',256*1024)
                metrics=json.loads(data)
                trace=json.loads(download(client,ref,version,prefix+'RESULT.json',256*1024))
                if (trace['kind'],trace['optimizer_updates'],trace['best_epoch'])!=(
                    kind,experiment['optimizer_updates'],experiment['best_epoch']
                ) or trace['checkpoints']!=experiment['checkpoints']:
                    raise ValueError('Experiment report differs from the saved result')
                directory=output/kind
                atomic_json(directory/'BEST_METRICS.json',metrics)
                atomic_json(directory/'RESULT.json',trace)
                if kind=='pilot_geometry_refine':
                    baseline=json.loads(download(client,ref,version,prefix+'BASELINE.json',256*1024))
                    atomic_json(directory/'BASELINE.json',baseline)
                rows.append({'name':prefix+'BEST_METRICS.json','status':'COLLECTED',
                             'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    receipt={'schema':'topface-exact-compact-collection-v1.1','ref':ref,'version':version,'run_id':run_id,
        'observed_utc':datetime.now(timezone.utc).isoformat(),'members':rows,
        'raw_archive_downloaded_to_host':False,'automatic_replay':False}
    atomic_json(output/'COLLECTION.json',receipt);print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);p.add_argument('--version',type=int,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--publisher-route',action='store_true')
    p.add_argument('--similarity',action='store_true')
    p.add_argument('--training',action='store_true')
    try:collect(**vars(p.parse_args()))
    except Exception as error:
        raise SystemExit(f'Bounded output collection failed: {type(error).__name__}') from None
