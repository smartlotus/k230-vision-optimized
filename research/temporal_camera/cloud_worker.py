"""Kaggle-only producer; durable publisher uses the original saved archive."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cloud_store import archive_directory, sha, atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run-id',required=True);ap.add_argument('--input-sha',required=True)
    ap.add_argument('--wall-seconds',type=int,default=2400)
    ap.add_argument('--encoder',default='baseline');ap.add_argument('--all-endpoints',action='store_true')
    args=ap.parse_args()
    if Path.cwd()!=Path('/kaggle/working'):raise ValueError('Cloud working directory required')
    outputs=Path('/kaggle/working')/args.run_id;outputs.mkdir(exist_ok=False)
    for name in ('INPUT_MANIFEST.json','data/manifest.jsonl','data/QA.json','data/RENDER_IDENTITY.json'):
        src=args.root/name
        shutil.copy2(src,outputs/src.name)
    source=outputs/'source';source.mkdir()
    for path in args.root.glob('*.py'):shutil.copy2(path,source/path.name)
    begin=time.monotonic()
    with (outputs/'train.log').open('x') as log:
        command=[sys.executable,str(args.root/'train.py'),'--data',str(args.root/'data'),
              '--out',str(outputs/'models'),'--run-id',args.run_id,'--epochs','60','--max-seconds',str(args.wall_seconds)]
        command.extend(['--encoder',args.encoder])
        if args.all_endpoints:command.append('--all-endpoints')
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.wall_seconds+100)
    state={'run_id':args.run_id,'input_manifest_sha256':args.input_sha,
           'compute_status':'PASS' if result.returncode==0 else 'FAILED',
           'train_exit_code':result.returncode,'elapsed_seconds':time.monotonic()-begin,
           'publication':'PENDING_INDEPENDENT_CLOUD_PUBLISHER','automatic_replay':False}
    atomic_json(outputs/'COMPUTE.json',state)
    archive=Path('/kaggle/working')/(args.run_id+'.tar.gz')
    archive_directory(outputs,archive)
    state['archive']={'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}
    atomic_json(Path('/kaggle/working')/'RESULT.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
