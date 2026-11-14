"""Upload one verified private Kaggle input and record the create result."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from cloud_store import atomic_json, sha
from native_job import verify_inputs


def create(staging, frozen, token_file):
    staging = staging.resolve()
    intent = json.loads((frozen/'SUBMISSION_INTENT.json').read_text(encoding='utf8'))
    data = json.loads((staging/'dataset-metadata.json').read_text(encoding='utf8'))
    if data['id'] != intent['dataset'] or intent['dataset_version'] != 1:
        raise ValueError('Dataset identity differs from its frozen intent')
    manifest = verify_inputs(staging)
    if (manifest['run_id'] != intent['run_id'] or
            sha(staging/'INPUT_MANIFEST.json') != intent['input_manifest_sha256']):
        raise ValueError('The staging manifest differs from its submitted identity')
    members = {row['path'] for row in manifest['files']}
    if any('/' in name or '\\' in name for name in members):
        raise ValueError('Only explicitly listed root files can be uploaded with this command')
    if set(path.name for path in staging.iterdir()) != members|{'INPUT_MANIFEST.json','dataset-metadata.json'}:
        raise ValueError('Unexpected files in the private dataset staging directory')
    attempt = frozen/'DATASET_SAVE_ATTEMPT.json'
    if attempt.exists():
        raise FileExistsError('Reconcile the previous dataset creation before submitting again')
    payload = {'ref':data['id'],'version':1,'private':True,'manifest_sha256':intent['input_manifest_sha256'],
               'observed_utc':datetime.now(timezone.utc).isoformat()}
    with attempt.open('x', encoding='utf8') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token_file.read_text(encoding='utf8').strip()
    result = subprocess.run([sys.executable,'-m','kaggle','datasets','create','-p',str(staging),
                             '-q','-t','-r','skip'],capture_output=True,text=True,
                            encoding='utf8',errors='replace',env=env)
    if result.returncode or 'Your private Dataset is being created' not in result.stdout:
        payload.update(status='RECONCILE_DATASET',exit_code=result.returncode)
        atomic_json(frozen/'DATASET_SAVE_RESPONSE.json',payload)
        raise RuntimeError('Dataset create did not confirm success. Check its exact reference.')
    payload.update(status='CREATING',exit_code=0)
    atomic_json(frozen/'DATASET_SAVE_RESPONSE.json',payload)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('staging','frozen','token-file'):
        parser.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(create(**vars(parser.parse_args())),indent=2))
