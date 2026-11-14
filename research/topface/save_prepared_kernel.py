"""One Quick Save with persisted intent; never launches scientific compute."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

from cloud_store import atomic_json, sha


def main(kernel_dir, frozen_dir, token_file):
    kernel_dir,frozen_dir=Path(kernel_dir),Path(frozen_dir)
    intent=json.loads((frozen_dir/'SUBMISSION_INTENT.json').read_text(encoding='utf8'))
    if sha(kernel_dir/'01_topface_simulation.ipynb')!=intent['notebook_sha256']:
        raise ValueError('Notebook differs from frozen intent')
    attempt=frozen_dir/'KERNEL_SAVE_ATTEMPT.json'
    if attempt.exists():raise RuntimeError('Previous Save attempt requires read-only reconciliation')
    os.environ['KAGGLE_API_TOKEN']=Path(token_file).read_text(encoding='utf8').strip()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    atomic_json(attempt,{'operation':'QUICK_SAVE','observed_utc':datetime.now(timezone.utc).isoformat(),
        'ref':intent['kernel'],'source_sha256':intent['notebook_sha256'],'session_timeout_seconds':10800,
        'scientific_execution':False,'automatic_retry':False})
    # Inspected kaggle2.2.5 kernels_push -> SDK save_kernel -> requests.send:
    # a single Save call, no mutation retry decorator.
    try:
        response=api.kernels_push(str(kernel_dir),timeout='10800',acc='NvidiaTeslaT4',no_run=True)
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
    main(**vars(p.parse_args()))
