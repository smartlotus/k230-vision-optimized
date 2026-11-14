"""Archive the already-produced LOCAL Blender qualification, not cloud output.

This input migration is separate from the cloud-first rule for newly produced
Kaggle artifacts. Full remote bytes are hashed as a stream, never saved twice.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from cloud_store import archive_directory, atomic_json, sha

FOLDER='1MwltNmaI24HjvcxnG1w608qui42RAPKf'
REMOTE='gdrive:k230-topfacenet-20261114'


def main(root, archive, receipt, rclone):
    root,archive,receipt=map(Path,(root,archive,receipt))
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf8'))
        if archive.exists() and sha(archive)==old['sha256']:
            print(json.dumps(old));return
        raise RuntimeError('Existing acceptance must be reconciled, not overwritten')
    summary=json.loads((root/'AUDIT_V11.json').read_text())
    if summary['status']!='PASS' or summary['images']!=100:
        raise ValueError('Only the 100-scene qualified Smoke is accepted here')
    repo=Path(__file__).resolve().parents[2]
    gate=repo/'docs/topface/v11/reports/SMOKE_HUMAN_REVIEW.json'
    if json.loads(gate.read_text(encoding='utf8'))['status']!='PASS':
        raise ValueError('Human review is required')
    if not archive.exists():
        shutil.copy2(gate,root/'SMOKE_HUMAN_REVIEW.json')
        atomic_json(root/'PROVENANCE_SCOPE.json',{
            'producer':'LOCAL Blender 4.2.9 CPU','not_kaggle':True,'images':100,
            'source_freeze_before_generation':False,
            'source_identity_limit':'Original Smoke predates full worker freeze. Raw per-scene camera/object/seed metadata and actual render receipts are retained; current source hashes are not asserted to prove original render source.',
            'audit_version_sha256':sha(repo/'research/topface/audit_v11.py'),
            'human_review_sha256':sha(gate),'model_training_started':False})
        archive_directory(root,archive)
    digest,size=sha(archive),archive.stat().st_size
    name=f'k230-v11-local-smoke100-{digest[:16]}.tar.gz'
    def command(*args):
        result=subprocess.run([str(rclone),*args,'--max-duration','120s'],capture_output=True,check=False)
        if result.returncode:raise RuntimeError('Bounded rclone operation failed; credential contents are not logged')
        return result.stdout
    folders=json.loads(command('lsjson','gdrive:','--dirs-only','--max-depth','1'))
    matches=[r for r in folders if r.get('Name')=='k230-topfacenet-20261114']
    if len(matches)!=1 or matches[0].get('ID')!=FOLDER:
        raise ValueError('Drive folder identity mismatch')
    command('copyto',str(archive),REMOTE+'/'+name,'--immutable')
    objects=json.loads(command('lsjson',REMOTE,'--max-depth','1'))
    objects=[r for r in objects if r.get('Name')==name]
    if len(objects)!=1 or objects[0].get('Size')!=size or not objects[0].get('ID'):
        raise ValueError('Drive object identity/size is ambiguous')
    returned=hashlib.sha256();count=0
    child=subprocess.Popen([str(rclone),'cat',REMOTE+'/'+name,'--max-duration','120s'],
                           stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        while chunk:=child.stdout.read(1024**2):
            count+=len(chunk);returned.update(chunk)
            if count>size:raise ValueError('Remote content exceeds frozen archive')
        if child.wait(timeout=10):raise RuntimeError('Remote full-byte readback failed')
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
        child.stdout.close()
    if (count,returned.hexdigest())!=(size,digest):
        raise ValueError('Full-byte readback digest mismatch')
    result={'schema':'topface-local-qualification-migration-v1.1',
        'status':'FULL_BYTE_SHA256_PASS','observed_utc':datetime.now(timezone.utc).isoformat(),
        'run_id':'LOCAL_NATIVE_BLENDER_SMOKE100','producer':'local CPU, not Kaggle',
        'folder_id':FOLDER,'file_id':objects[0]['ID'],'name':name,'bytes':size,'sha256':digest,
        'member_manifest_sha256':sha(root/'MEMBERS.json'),'images':100,
        'instances':summary['instances'],'human_review':'PASS','automatic_replay':False,
        'provider_readback':'FULL_BYTE_SHA256_PASS'}
    atomic_json(receipt,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--rclone',type=Path,required=True);main(**vars(p.parse_args()))
