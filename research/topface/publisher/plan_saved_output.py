"""Turn a saved Kaggle result and terminal observation into a publisher plan."""
import argparse
import json
from pathlib import Path
import re

from cloud_store import atomic_json, sha


def plan(frozen, observation, result):
    intent = json.loads((frozen/'SUBMISSION_INTENT.json').read_text(encoding='utf8'))
    provider = json.loads(observation.read_text(encoding='utf8'))
    saved = json.loads(result.read_text(encoding='utf8'))
    if provider['status'] != 'COMPLETE' or provider['ref'] != intent['kernel']:
        raise ValueError('Expected a completed exact-version Kaggle output')
    if (saved['run_id'], saved['input_manifest_sha256']) != (intent['run_id'], intent['input_manifest_sha256']):
        raise ValueError('Saved output differs from the frozen execution identity')
    archive = saved['archive']
    run_id = intent['run_id']
    if (archive['name'] != run_id+'.tar.gz' or not re.fullmatch(r'[a-f0-9]{64}',archive['sha256']) or
            not 0 < archive['bytes'] <= 1024**3):
        raise ValueError('Saved archive metadata differs from the publisher contract')
    notebook = frozen/intent['notebook_file']
    digest = sha(notebook)
    if digest != intent['notebook_sha256']:
        raise ValueError('Frozen notebook differs from the submission record')
    return {'run_id':run_id,'source_archive_path':archive['name'],
            'archive':{**archive,'name':f"{run_id}-{archive['sha256'][:16]}.tar.gz"},
            'prep':{'ref':provider['ref'],'kernel_id':provider['kernel_id'],'version':provider['version'],
                    'provider_status':provider['status'],'source_sha256':digest},
            'source_notebook':notebook.as_posix()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('frozen','observation','result','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Publisher plan already exists')
    atomic_json(args.output,plan(args.frozen,args.observation,args.result))
