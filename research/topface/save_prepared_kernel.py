"""Submit one prepared Kaggle version with a persisted intent."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

from cloud_store import atomic_json, sha


def main(kernel_dir, frozen_dir, token_file,execute=False):
    kernel_dir,frozen_dir=Path(kernel_dir),Path(frozen_dir)
    intent=json.loads((frozen_dir/'SUBMISSION_INTENT.json').read_text(encoding='utf8'))
    metadata=json.loads((kernel_dir/'kernel-metadata.json').read_text(encoding='utf8'))
    notebook=intent.get('notebook_file','01_topface_simulation.ipynb')
    if metadata['code_file']!=notebook or metadata['id']!=intent['kernel']:
        raise ValueError('Kernel metadata differs from frozen intent')
    if metadata!=json.loads((frozen_dir/'kernel-metadata.json').read_text(encoding='utf8')):
        raise ValueError('Kernel source mounts or settings differ from frozen metadata')
    if sha(kernel_dir/notebook)!=intent['notebook_sha256']:
        raise ValueError('Notebook differs from frozen intent')
    if execute and (intent.get('execution_mode')!='API_SAVE_AND_RUN_ALL' or intent.get('publisher_admission_status')!='QUALIFIED_AND_AUTHORIZED'):
        raise ValueError('Automatic compute requires qualified authorized independentpublisher admission')
    attempt=frozen_dir/'KERNEL_SAVE_ATTEMPT.json'
    if attempt.exists():raise RuntimeError('Previous Save attempt requires read-only reconciliation')
    os.environ['KAGGLE_API_TOKEN']=Path(token_file).read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    atomic_json(attempt,{'operation':'SAVE_AND_RUN_ALL' if execute else 'QUICK_SAVE','observed_utc':datetime.now(timezone.utc).isoformat(),
        'ref':intent['kernel'],'source_sha256':intent['notebook_sha256'],
        'session_timeout_seconds':intent.get('provider_max_seconds',10800),
        'scientific_execution':execute,'automatic_retry':False})
    # Inspected kaggle2.2.5 kernels_push -> SDK save_kernel -> requests.send:
    # a single Save call, no mutation retry decorator.
    try:
        response=api.kernels_push(str(kernel_dir),timeout=str(intent.get('provider_max_seconds',10800)),
            acc='NvidiaTeslaT4' if metadata.get('enable_gpu') else None,no_run=not execute)
        result=type(response).to_dict(response)
        atomic_json(frozen_dir/'KERNEL_SAVE_RESPONSE.json',result)
        print(json.dumps(result))
    except Exception as error:
        atomic_json(frozen_dir/'KERNEL_SAVE_UNCERTAIN.json',{'status':'RECONCILE_ORIGINAL_REF',
            'error_type':type(error).__name__,'automatic_retry':False})
        raise RuntimeError('Save failed or response uncertain; reconcile exact ref before any retry') from None


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel-dir',type=Path,required=True)
    p.add_argument('--frozen-dir',type=Path,required=True);p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    main(**vars(p.parse_args()))
