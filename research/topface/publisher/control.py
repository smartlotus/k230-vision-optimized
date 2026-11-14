"""Authorized ACT-style single-send/reconcile publisher host controller.

Raw archives never traverse this host. Sensitive recovery is CurrentUser-DPAPI
encrypted; public binding/receipt contains no URLs, access token or secret value.
"""
import argparse
import configparser
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/topface'))
from cloud_store import Drive,API,atomic_json
from publisher.protected_state import save as protect,load as unprotect
from publisher.ci_transfer import parse_spec,source_endpoint

REPO='smartlotus/k230-vision-optimized'
FOLDER='1MwltNmaI24HjvcxnG1w608qui42RAPKf'
WORKFLOW='.github/workflows/topface_archive_transfer.yml'


def command(program,*args,data=None,timeout=90):
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    for name in ('KAGGLE_API_TOKEN','DOASTRA_GDRIVE_OAUTH_JSON','TOPFACE_TRANSFER_SPEC'):env.pop(name,None)
    result=subprocess.run([program,*args],input=data,capture_output=True,text=True,encoding='utf8',
        cwd=ROOT,timeout=timeout,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise RuntimeError(program+' operation failed; reconcile original binding (no provider exception strings)')
    return result.stdout.strip()


def authorized():
    path=ROOT/'docs/topface/v11/cloud/PUBLISHER_AUTHORIZATION.json'
    value=json.loads(path.read_text(encoding='utf8'))
    if value.get('status')!='EXPLICIT_USER_APPROVED' or value.get('repository')!=REPO or value.get('drive_folder_id')!=FOLDER:
        raise ValueError('Explicit matching user publisher authorization required')
    if json.loads(command('gh','repo','view',REPO,'--json','isPrivate'))['isPrivate'] is not True:
        raise ValueError('Existing authorized repository is not private')


def drive_local():
    import requests
    config=configparser.ConfigParser(interpolation=None)
    config.read(Path(os.environ['APPDATA'])/'rclone/rclone.conf',encoding='utf8')
    remote=config['gdrive']
    if remote.get('type')!='drive' or remote.get('scope')!='drive.file':
        raise ValueError('Drive OAuth scope changed from explicitly approved drive.file')
    token=json.loads(remote['token'])
    drive=object.__new__(Drive)
    drive.oauth={'client_id':remote['client_id'],'client_secret':remote['client_secret'],'refresh_token':token['refresh_token']}
    drive.folder_id=FOLDER;drive.session=requests.Session();drive.token='';drive.expiry=0.
    drive.deadline=time.monotonic()+600
    drive.verify_folder()
    if drive.expiry<time.time()+1200:raise ValueError('Short access token lifetime insufficient for finite publisher')
    return drive


def protected_path(run_id):
    if not re.fullmatch(r'k230-[a-z0-9-]{8,100}',run_id):raise ValueError('Unsafe protected recovery run identity')
    path=ROOT/'.ops/local/topface-publisher-private'/ (run_id+'.dpapi')
    # Refuse sensitive staging if repository ignore rules no longer protect it.
    if subprocess.run(['git','check-ignore','--quiet',str(path)],cwd=ROOT).returncode:
        raise ValueError('Private encrypted recovery path is not Git ignored')
    return path


def source_url(prep,file_path,token_file,source_notebook):
    ref,version=prep['ref'],prep['version']
    source_notebook=(ROOT/source_notebook).resolve()
    if not source_notebook.is_relative_to(ROOT) or hashlib.sha256(source_notebook.read_bytes()).hexdigest()!=prep['source_sha256']:
        raise ValueError('Frozen local source SHA differs from transfer identity')
    expected=json.loads(source_notebook.read_text(encoding='utf8'))
    def cells(notebook):
        return [(c['cell_type'],''.join(c['source']) if isinstance(c['source'],list) else c['source']) for c in notebook['cells']]
    token=Path(token_file).read_text(encoding='utf8').strip()
    previous=os.environ.get('KAGGLE_API_TOKEN');os.environ['KAGGLE_API_TOKEN']=token
    try:
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest,ApiGetKernelSessionStatusRequest,ApiGetKernelRequest
        owner,slug=ref.split('/')
        with api.build_kaggle_client() as client:
            get=ApiGetKernelRequest();get.user_name=owner;get.kernel_slug=slug;get.version_label=f'v{version}'
            saved=client.kernels.kernels_api_client.get_kernel(get)
            if saved.metadata.id!=prep['kernel_id'] or not saved.metadata.is_private:
                raise ValueError('Exact source kernel ID/privacy differs')
            actual=json.loads(saved.blob.source)
            if cells(expected)!=cells(actual):raise ValueError('Exact saved source cells differ from reviewed SHA-bound input')
            status_request=ApiGetKernelSessionStatusRequest();status_request.user_name=owner
            status_request.kernel_slug=slug;status_request.version_label=f'v{version}'
            if client.kernels.kernels_api_client.get_kernel_session_status(status_request).status.name!='COMPLETE':
                raise ValueError('Exact source Kaggle session is not COMPLETE')
            page=None
            for _ in range(100):
                request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
                request.version_label=f'v{version}';request.page_size=1000;request.page_token=page
                response=client.kernels.kernels_api_client.list_kernel_session_output(request)
                matches=[item for item in response.files if item.file_name==file_path]
                if matches:
                    if len(matches)!=1:raise ValueError('Ambiguous archive output identity')
                    url=matches[0].url;source_endpoint(url);return url
                page=response.next_page_token
                if not page:break
        raise ValueError('Exact saved source archive missing')
    finally:
        if previous is None:os.environ.pop('KAGGLE_API_TOKEN',None)
        else:os.environ['KAGGLE_API_TOKEN']=previous


def folder_for(run_id):
    if not re.fullmatch(r'k230-[a-z0-9-]{8,100}',run_id):raise ValueError('Safe K230 run identity required')
    return ROOT/'docs/topface/v11/cloud/publisher'/run_id


def validate_binding(binding,spec=None):
    if binding.get('folder_id')!=FOLDER:raise ValueError('Binding destination differs')
    suffix=binding['secret_name'].removeprefix('TOPFACE_TRANSFER_')
    if not re.fullmatch(r'\d{8,30}',suffix) or binding['branch']!='ops/topface-transfer-'+suffix:
        raise ValueError('Binding Secret/branch identity differs')
    if spec is not None and any(spec[key]!=binding[key] for key in ('run_id','archive','prep','drive_file_id','folder_id')):
        raise ValueError('Protected envelope differs from original binding')
    head,source=binding['head'],binding['publisher_source_commit']
    if not all(re.fullmatch(r'[a-f0-9]{40}',h) for h in (head,source)):raise ValueError('Invalid publisher commit identity')
    if command('git','show','-s','--format=%B',head)!=suffix:
        raise ValueError('Publisher head message differs from Secret binding')
    if command('git','rev-list','--parents','-n','1',head).split()!=[head,source]:
        raise ValueError('Publisher exact parent differs')
    if command('git','rev-parse',head+'^{tree}')!=command('git','rev-parse',source+'^{tree}'):
        raise ValueError('Publisher tree differs from reviewed source revision')
    code=command('git','show',source+':research/topface/publisher/ci_transfer.py')
    # git show is decoded/re-encoded by command(text), so compare canonical
    # source lines here; the actual checkout retains source Git blob bytes.
    local=(ROOT/'research/topface/publisher/ci_transfer.py').read_text(encoding='utf8').strip()
    if code!=local:raise ValueError('Installed publisher differs from reviewed implementation')
    workflow=command('git','show',source+':'+WORKFLOW)
    if workflow!=(ROOT/'research/topface/publisher/workflow.yml').read_text(encoding='utf8').strip():
        raise ValueError('Installed workflow differs from reviewed template')


@contextmanager
def run_lock(run_id):
    path=protected_path(run_id).with_suffix('.lock');path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,json.dumps({'pid':os.getpid(),'run_id':run_id,'automatic_lock_reclaim':False}).encode())
        yield
    finally:
        os.close(descriptor);path.unlink()


def prepare(plan_path,token_file,source_commit):
    authorized();plan=json.loads(Path(plan_path).read_text(encoding='utf8'));run_id=plan['run_id']
    folder=folder_for(run_id);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'BINDING.json').exists():raise ValueError('Existing transfer binding must be reconciled, not replaced')
    if not re.fullmatch(r'[a-f0-9]{40}',source_commit):raise ValueError('Exact installed publisher source commit required')
    command('git','cat-file','-e',source_commit+':'+WORKFLOW)
    command('git','cat-file','-e',source_commit+':research/topface/publisher/ci_transfer.py')
    archive=plan['archive'];prep=plan['prep']
    if prep['provider_status']!='COMPLETE':raise ValueError('Verified completed source required')
    url=source_url(prep,plan['source_archive_path'],token_file,plan['source_notebook'])
    drive=drive_local()
    # Reserve identity and local attempt before any mutable Drive upload request.
    with drive.request('GET',API+'/generateIds',params={'count':1,'space':'drive','type':'files'}) as response:
        file_id,=response.json()['ids']
    suffix=str(time.time_ns());branch='ops/topface-transfer-'+suffix;secret='TOPFACE_TRANSFER_'+suffix
    tree=command('git','rev-parse',source_commit+'^{tree}')
    head=command('git','commit-tree',tree,'-p',source_commit,'-m',suffix)
    binding={'schema':'topface-cloud-transfer-binding-v1.1','run_id':run_id,'archive':archive,
        'prep':prep,'drive_file_id':file_id,'folder_id':FOLDER,'head':head,'branch':branch,'secret_name':secret,
        'publisher_source_commit':source_commit,'created_utc':datetime.now(timezone.utc).isoformat(),
        'automatic_replay':False,'source_archive_path':plan['source_archive_path']}
    atomic_json(folder/'BINDING.json',binding)
    try:
        with drive.request('POST','https://www.googleapis.com/upload/drive/v3/files',
            params={'uploadType':'resumable','fields':'id,name,size,parents,sha256Checksum'},
            headers={'X-Upload-Content-Length':str(archive['bytes']),'X-Upload-Content-Type':'application/gzip'},
            json={'id':file_id,'name':archive['name'],'parents':[FOLDER],
                'appProperties':{'project':'k230-topface','run_id':run_id,'sha256':archive['sha256']}}) as response:
            upload=response.headers['Location']
        spec={'run_id':run_id,'archive':archive,'prep':prep,'folder_id':FOLDER,'drive_file_id':file_id,
            'source_url':url,'upload_url':upload,'drive_access_token':drive.token,'drive_expiry_epoch':drive.expiry}
        raw=json.dumps(spec);parse_spec(raw)
        protect(protected_path(run_id),raw.encode('utf8'),run_id)
        atomic_json(folder/'PREPARED.json',{'status':'PROTECTED_ENVELOPE_READY','run_id':run_id,
            'oauth_scope':'drive.file','long_lived_credentials_sent':False,'local_raw_archive_downloaded':False})
    except Exception as error:
        atomic_json(folder/'PREPARE_UNCERTAIN.json',{'status':'RECONCILE_RESERVED_DRIVE_ID','error_type':type(error).__name__,
            'drive_file_id':file_id,'automatic_retry':False})
        raise RuntimeError('Transfer preparation failed; reconcile original Drive identity') from None
    return binding


def dispatch(run_id):
    authorized();folder=folder_for(run_id);binding=json.loads((folder/'BINDING.json').read_text(encoding='utf8'))
    validate_binding(binding)
    remote=command('git','ls-remote','origin','refs/heads/'+binding['branch'])
    runs=json.loads(command('gh','run','list','--repo',REPO,'--branch',binding['branch'],'--commit',binding['head'],
        '--limit','4','--json','headSha,headBranch,event'))
    if remote or runs:
        if remote and remote.split()!=[binding['head'],'refs/heads/'+binding['branch']]:raise ValueError('Remote transfer branch mismatch')
        if len(runs)>1 or any(r['headSha']!=binding['head'] or r['headBranch']!=binding['branch'] or r['event']!='push' for r in runs):
            raise ValueError('Ambiguous existing transfer run')
        return {'status':'ORIGINAL_DISPATCH_RECONCILED','automatic_replay':False}
    raw=unprotect(protected_path(run_id),run_id).decode('utf8');spec=parse_spec(raw);validate_binding(binding,spec)
    atomic_json(folder/'DISPATCH_ATTEMPT.json',{'head':binding['head'],'branch':binding['branch'],
        'secret_name':binding['secret_name'],'automatic_retry':False})
    command('gh','secret','set',binding['secret_name'],'--repo',REPO,data=raw)
    # The exact unique push is the final trigger. A lost response is reconciled
    # against this same head/branch, never used to repeat compute or allocate IDs.
    command('git','push','origin',binding['head']+':refs/heads/'+binding['branch'])
    atomic_json(folder/'DISPATCHED.json',{'head':binding['head'],'branch':binding['branch'],'status':'SENT_ONCE'})
    return {'status':'SENT_ONCE','head':binding['head'],'branch':binding['branch']}


def observe(run_id):
    authorized();folder=folder_for(run_id);binding=json.loads((folder/'BINDING.json').read_text(encoding='utf8'))
    accepted=folder/'TRANSFER_RECEIPT.json'
    if accepted.exists():return json.loads(accepted.read_text(encoding='utf8'))
    validate_binding(binding)
    runs=json.loads(command('gh','run','list','--repo',REPO,'--branch',binding['branch'],'--commit',binding['head'],
        '--limit','4','--json','databaseId,status,conclusion,headSha,headBranch,event'))
    if not runs:return {'status':'AWAITING_ORIGINAL_RUN','automatic_replay':False}
    if len(runs)!=1:raise ValueError('Multiple workflow identities require reconciliation')
    run=runs[0]
    if (run['headSha'],run['headBranch'],run['event'])!=(binding['head'],binding['branch'],'push'):
        raise ValueError('Workflow original identity differs')
    atomic_json(folder/'OBSERVATION.json',run)
    if run['status']!='completed':return run
    secrets=json.loads(command('gh','secret','list','--repo',REPO,'--json','name'))
    if any(s['name']==binding['secret_name'] for s in secrets):command('gh','secret','delete',binding['secret_name'],'--repo',REPO)
    remaining=json.loads(command('gh','secret','list','--repo',REPO,'--json','name'))
    if any(s['name']==binding['secret_name'] for s in remaining):raise RuntimeError('Temporary Secret cleanup unconfirmed')
    atomic_json(folder/'SECRET_CLEANUP.json',{'status':'DELETED_AFTER_TERMINAL','secret_name':binding['secret_name'],
        'github_run_id':run['databaseId'],'secret_value_logged':False})
    if run['conclusion']!='success':return {'status':'TRANSFER_FAILED_TERMINAL_SECRET_REMOVED','github_run_id':run['databaseId'],'automatic_replay':False}
    jobs=json.loads(command('gh','run','view',str(run['databaseId']),'--repo',REPO,'--json','jobs'))['jobs']
    if len(jobs)!=1 or jobs[0]['name']!='transfer' or jobs[0]['conclusion']!='success':raise ValueError('Unexpected publisher jobs')
    log=command('gh','api',f'repos/{REPO}/actions/jobs/{jobs[0]["databaseId"]}/logs')
    matches=re.findall(r'TOPFACE_TRANSFER_RECEIPT (\{[^\r\n]*\})',log)
    if len(matches)!=1:raise ValueError('Unique compact publisher receipt missing')
    result=json.loads(matches[0]);uploaded=result['uploaded']
    if result['run_id']!=run_id or result['prep']!=binding['prep'] or result['folder_id']!=FOLDER or result['automatic_replay'] is not False:
        raise ValueError('Publisher receipt identity mismatch')
    if result['status']!='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or uploaded['file_id']!=binding['drive_file_id'] or uploaded['provider_readback']!='FULL_BYTE_SHA256_PASS':
        raise ValueError('Publisher full-byte acceptance missing')
    if any(uploaded[k]!=binding['archive'][k] for k in ('name','bytes','sha256')):raise ValueError('Published archive differs')
    drive=drive_local()
    with drive.request('GET',API+'/'+binding['drive_file_id'],params={'fields':'id,name,size,parents,trashed,sha256Checksum'}) as response:meta=response.json()
    if meta.get('id')!=uploaded['file_id'] or meta.get('name')!=uploaded['name'] or meta.get('parents')!=[FOLDER] or int(meta.get('size',-1))!=uploaded['bytes'] or meta.get('trashed'):
        raise ValueError('Independent Drive metadata does not match full-byte cloud receipt')
    result.update(github={'run_id':run['databaseId'],'head':binding['head'],'branch':binding['branch']},
        secret_cleanup='CONFIRMED_DELETED',independent_metadata_match=True,raw_archive_downloaded_to_host=False)
    atomic_json(accepted,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','dispatch','observe'])
    parser.add_argument('--plan',type=Path);parser.add_argument('--token-file',type=Path)
    parser.add_argument('--source-commit');parser.add_argument('--run-id');args=parser.parse_args()
    run_id=json.loads(args.plan.read_text(encoding='utf8'))['run_id'] if args.operation=='prepare' else args.run_id
    with run_lock(run_id):
        if args.operation=='prepare':result=prepare(args.plan,args.token_file,args.source_commit)
        elif args.operation=='dispatch':result=dispatch(args.run_id)
        else:result=observe(args.run_id)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        # ConfigParser/provider exceptions may embed original credential lines,
        # signed URLs or headers. Never emit exception strings or tracebacks.
        print(json.dumps({'status':'PUBLISHER_OPERATION_NEEDS_RECONCILIATION','error_type':type(error).__name__,
            'automatic_retry':False}),file=sys.stderr)
        raise SystemExit(1) from None
