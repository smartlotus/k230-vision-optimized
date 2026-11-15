"""Refresh one terminal publisher attempt, retaining its archive, file and session."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_store import API,atomic_json,sha
from publisher.control import (authorized,command,drive_local,folder_for,protect,protected_path,
    run_lock,source_url,unprotect,validate_binding,REPO)
from publisher.ci_transfer import parse_spec


def recover(run_id,token_file,source_commit):
    authorized()
    folder=folder_for(run_id)
    binding=json.loads((folder/'BINDING.json').read_text(encoding='utf8'))
    cached=json.loads(unprotect(protected_path(run_id),run_id).decode('utf8'))
    validate_binding(binding,cached,compare_local_source=False)
    runs=json.loads(command('gh','run','list','--repo',REPO,'--branch',binding['branch'],
        '--commit',binding['head'],'--limit','4','--json','databaseId,status,conclusion,headSha,headBranch,event'))
    if (len(runs)!=1 or runs[0]['status']!='completed' or runs[0]['conclusion']=='success' or
            (runs[0]['headSha'],runs[0]['headBranch'],runs[0]['event'])!=(binding['head'],binding['branch'],'push')):
        raise ValueError('Recovery requires one confirmed terminal unsuccessful transport attempt')
    secrets=json.loads(command('gh','secret','list','--repo',REPO,'--json','name'))
    if any(row['name']==binding['secret_name'] for row in secrets):
        raise ValueError('Original terminal Secret must be cleaned before preparing recovery')
    # The immutable plan lives with the frozen producer, not the transport.
    plans=list((folder.parents[1]).glob('*/PUBLISHER_PLAN.json'))
    matches=[path for path in plans if json.loads(path.read_text(encoding='utf8'))['run_id']==run_id]
    if len(matches)!=1:
        raise ValueError('Expected the original immutable publisher plan')
    original_plan=json.loads(matches[0].read_text(encoding='utf8'))
    if any(original_plan[key]!=binding[key] for key in ('run_id','archive','prep','source_archive_path')):
        raise ValueError('Original publisher plan differs from recovery archive identity')
    drive=drive_local()
    exists=False
    try:
        with drive.request('GET',API+'/'+binding['drive_file_id'],
            params={'fields':'id,name,size,parents,trashed,sha256Checksum'}) as response:metadata=response.json()
        if (metadata.get('id')!=binding['drive_file_id'] or metadata.get('name')!=binding['archive']['name'] or
                metadata.get('parents')!=[binding['folder_id']] or metadata.get('trashed') or
                int(metadata.get('size',-1))!=binding['archive']['bytes'] or
                metadata.get('sha256Checksum',binding['archive']['sha256'])!=binding['archive']['sha256']):
            raise ValueError('Existing reserved object differs from the original archive')
        exists=True
    except Exception as error:
        if getattr(getattr(error,'response',None),'status_code',None)!=404:
            raise
    confirmed=binding['archive']['bytes'] if exists else None
    if not exists:
        with drive.request('PUT',cached['upload_url'],data=b'',
                headers={'Content-Length':'0','Content-Range':'bytes */'+str(binding['archive']['bytes'])}) as response:
            matched=re.fullmatch(r'bytes=0-(\d+)',response.headers.get('Range',''))
            if response.status_code!=308:
                raise ValueError('Original upload session needs separate terminal reconciliation')
            confirmed=int(matched[1])+1 if matched else 0
            if not 0<=confirmed<binding['archive']['bytes']:
                raise ValueError('Original session offset is outside the archive')
    url=source_url(binding['prep'],binding['source_archive_path'],token_file,original_plan['source_notebook'])
    suffix=str(time.time_ns())
    tree=command('git','rev-parse',source_commit+'^{tree}')
    head=command('git','commit-tree',tree,'-p',source_commit,'-m',suffix)
    refreshed={**binding,'head':head,'branch':'ops/topface-transfer-'+suffix,
        'secret_name':'TOPFACE_TRANSFER_'+suffix,'publisher_source_commit':source_commit,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'recovery_of_github_run_id':runs[0]['databaseId'],'confirmed_original_bytes_before_recovery':confirmed}
    spec={**cached,'source_url':url,'drive_access_token':drive.token,'drive_expiry_epoch':drive.expiry}
    parse_spec(json.dumps(spec));validate_binding(refreshed,spec)
    history=folder/'attempts'
    number=len(list(history.glob('attempt-*')))+1 if history.exists() else 1
    archive=history/f'attempt-{number:02d}'
    archive.mkdir(parents=True,exist_ok=False)
    names=('BINDING.json','DISPATCH_ATTEMPT.json','DISPATCHED.json','OBSERVATION.json',
           'PREPARED.json','SECRET_CLEANUP.json','RECOVERY_OBSERVATION.json')
    for name in names:
        source=folder/name
        if source.exists():
            shutil.copy2(source,archive/name)
            if sha(source)!=sha(archive/name):
                raise ValueError('Original transport history copy differs')
    private=protected_path(run_id)
    backup=private.with_name(private.stem+f'-attempt-{number:02d}.dpapi')
    with backup.open('xb') as stream:stream.write(private.read_bytes())
    public={'status':'RESUMABLE_RECOVERY_PREPARED','run_id':run_id,
        'original_github_run_id':runs[0]['databaseId'],'reserved_drive_file_id':binding['drive_file_id'],
        'archive':binding['archive'],'original_confirmed_bytes':confirmed,'existing_complete_object':exists,
        'same_resumable_session':True,'new_publisher_source_commit':source_commit,
        'science_replayed':False,'new_drive_object_allocated':False}
    atomic_json(archive/'RECOVERY.json',public)
    protect(private,json.dumps(spec).encode('utf8'),run_id)
    for name in names:
        path=folder/name
        if path.exists():
            if not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError('Transport history path is outside the original transfer folder')
            path.unlink()
    atomic_json(folder/'BINDING.json',refreshed)
    atomic_json(folder/'PREPARED.json',{'status':'PROTECTED_ENVELOPE_READY','run_id':run_id,
        'oauth_scope':'drive.file','long_lived_credentials_sent':False,'local_raw_archive_downloaded':False})
    atomic_json(folder/'RECOVERY_PREPARED.json',public)
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--token-file',type=Path,required=True);parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    try:
        with run_lock(args.run_id):
            print(json.dumps(recover(**vars(args)),indent=2))
    except Exception as error:
        print(json.dumps({'status':'RECOVERY_NEEDS_RECONCILIATION','error_type':type(error).__name__,
                          'automatic_retry':False}),file=sys.stderr)
        raise SystemExit(1) from None
