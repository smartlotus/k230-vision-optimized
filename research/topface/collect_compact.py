"""Exact-version JSON-only Kaggle collection, max256KiB each, no raw relay."""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path

from cloud_store import atomic_json


def collect(ref,version,run_id,output,token_file):
    if not run_id.replace('-','').isalnum() or version<1:raise ValueError('Explicit original run/version required')
    os.environ['KAGGLE_API_TOKEN']=Path(token_file).read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    owner,slug=ref.split('/');output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with api.build_kaggle_client() as client:
        for suffix in ('RESULT','DRIVE'):
            name=run_id+'-'+suffix+'.json';target=output/name
            if target.exists():
                value=json.loads(target.read_text(encoding='utf8'))
                if value.get('run_id')!=run_id:raise ValueError('Existing compact identity mismatch')
                rows.append({'name':name,'status':'ALREADY_COLLECTED','sha256':hashlib.sha256(target.read_bytes()).hexdigest()});continue
            request=ApiDownloadKernelOutputRequest();request.owner_slug=owner;request.kernel_slug=slug
            request.version_number=version;request.file_path=name
            try:
                with client.kernels.kernels_api_client.download_kernel_output(request) as response:
                    response.raise_for_status()
                    if int(response.headers.get('Content-Length','0'))>256*1024:
                        raise ValueError('Compact result exceeds local byte cap')
                    data=bytearray()
                    for chunk in response.iter_content(16384):
                        data.extend(chunk)
                        if len(data)>256*1024:raise ValueError('Compact result stream exceeds byte cap')
                value=json.loads(bytes(data))
                if value.get('run_id')!=run_id:raise ValueError('Downloaded compact run identity mismatch')
                temporary=target.with_suffix('.json.part');temporary.write_bytes(data);temporary.replace(target)
                rows.append({'name':name,'status':'COLLECTED','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            except Exception as error:
                # Source not saved yet is an observation failure, never evidence
                # to rerun science. No signed URL/exception strings in receipt.
                rows.append({'name':name,'status':'NOT_AVAILABLE_OR_FETCH_FAILED','error_type':type(error).__name__})
    receipt={'schema':'topface-exact-compact-collection-v1.1','ref':ref,'version':version,'run_id':run_id,
        'observed_utc':datetime.now(timezone.utc).isoformat(),'members':rows,
        'raw_archive_downloaded_to_host':False,'automatic_replay':False}
    atomic_json(output/'COLLECTION.json',receipt);print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);p.add_argument('--version',type=int,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True);collect(**vars(p.parse_args()))
