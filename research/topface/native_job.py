"""Finite V1.1 native-Blender Kaggle producer. No BlenderProc/bootstrap pip.

Cloud jobs use direct Drive admission or the authorized publisher.
Local qualification is limited to six frames.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

from cloud_store import Drive, archive_directory, atomic_json, sha

BLENDER_URL = 'https://download.blender.org/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz'
BLENDER_SHA256 = 'dfbc127a7d28f9c2175b23bf9d6701b2855f31eedfb391f9a6e60adb24572846'
BLENDER_NAME = 'blender-4.2.9-linux-x64'
MAX_DOWNLOAD = 400 * 1024**2


def verify_inputs(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'INPUT_MANIFEST.json').read_text(encoding='utf8'))
    for member in manifest['files']:
        path = (root/member['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError('Input member is missing or outside the frozen input')
        if (path.stat().st_size, sha(path)) != (member['bytes'], member['sha256']):
            raise ValueError('Frozen input bytes do not match manifest')
    return manifest


def execute(command, log, deadline, stage_cap):
    remaining = int(deadline-time.monotonic())
    if remaining < 30:
        raise TimeoutError('No registered execution budget remains')
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4')
    # Blender must use its own Python, not the notebook's Python path.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    with Path(log).open('wb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,env=env)
        child_deadline=time.monotonic()+min(stage_cap,remaining)
        cursor=0
        completed=0
        try:
            while True:
                try:
                    code=child.wait(timeout=min(10,max(.1,child_deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    code=None
                with Path(log).open('r',encoding='utf8',errors='replace') as tail:
                    tail.seek(cursor)
                    for line in tail:
                        if line.startswith('TOPFACE_PROGRESS '):
                            try:
                                progress=json.loads(line.split(' ',1)[1])
                            except json.JSONDecodeError:
                                continue
                            if progress['completed_frames']>=completed+25:
                                completed=progress['completed_frames']
                                print('TOPFACE_RENDER_HEARTBEAT',json.dumps(progress),flush=True)
                    cursor=tail.tell()
                if code is not None:
                    break
                if time.monotonic()>=child_deadline:
                    raise subprocess.TimeoutExpired(command,min(stage_cap,remaining))
        except BaseException:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
            raise
    if code:
        raise RuntimeError(f'Child process failed ({code}); inspect {Path(log).name}')


def install_blender(root, deadline):
    import requests
    archive = root/'blender-runtime.tar.xz'
    started = time.monotonic()
    total = 0
    with requests.get(BLENDER_URL, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with archive.open('xb') as stream:
            for chunk in response.iter_content(1024**2):
                if time.monotonic() > min(deadline, started+600):
                    raise TimeoutError('Blender download budget exhausted')
                total += len(chunk)
                if total > MAX_DOWNLOAD:
                    raise ValueError('Blender archive exceeds registered byte cap')
                stream.write(chunk)
    if sha(archive) != BLENDER_SHA256:
        raise ValueError('Official Blender archive digest mismatch')
    with tarfile.open(archive, 'r:xz') as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 2*1024**3:
            raise ValueError('Expanded Blender archive exceeds byte cap')
        # Official archive contains library symlinks; the data filter confines
        # destinations/links to the extraction root (Python >=3.11).
        bundle.extractall(root/'runtime', members=members, filter='data')
    return root/'runtime'/BLENDER_NAME/'blender'


def extract_cad(input_root, assets):
    assets.mkdir()
    with zipfile.ZipFile(input_root/'cad_bundle.bin') as bundle:
        members = [m for m in bundle.infolist() if not m.is_dir()]
        names = [Path(m.filename).name for m in members]
        if len(names) != len(set(names)) or sum(m.file_size for m in members) > 20*1024**2:
            raise ValueError('CAD archive inventory is ambiguous or too large')
        for member, name in zip(members, names):
            (assets/name).write_bytes(bundle.read(member))


def render(blender, inputs, assets, output, count, seed, device, log, deadline, qualification=False,start_index=0):
    command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
        '--python', str(inputs/'render_native.py'), '--', '--assets', str(assets),
        '--output', str(output), '--count', str(count), '--seed', str(seed),
        '--samples', '24', '--device', device, '--max-seconds', str(max(30, int(deadline-time.monotonic())-30))]
    if qualification:
        command.append('--qualification')
    elif (inputs/'SCENE_PLAN.json').is_file():
        command.extend(['--scene-plan',str(inputs/'SCENE_PLAN.json')])
    elif start_index:
        command.extend(['--start-index',str(start_index)])
    execute(command, log, deadline, 8100)


def main(input_root, work_root, run_id, count=1000, seed=20261115,
         folder_id='1MwltNmaI24HjvcxnG1w608qui42RAPKf', local_blender=None, max_seconds=9000):
    if not 1 <= count <= 1000 or not 120 <= max_seconds <= 9000:
        raise ValueError('Pilot count/time outside registered finite budget')
    if not run_id.replace('-', '').isalnum():
        raise ValueError('Unsafe run identity')
    input_root = Path(input_root).resolve()
    work_root = Path(work_root).resolve()
    local = local_blender is not None
    if local and count > 6:
        raise ValueError('This local worker qualification is limited to six frames')
    if not local and work_root != Path('/kaggle/working'):
        raise ValueError('Cloud producer must run in /kaggle/working')
    manifest = verify_inputs(input_root)
    render_config_path=input_root/'FULL_RENDER_CONFIG.json'
    publisher=render_config_path.is_file()
    start_index=0
    if publisher:
        config=json.loads(render_config_path.read_text(encoding='utf8'))
        if (config['run_id'],config['count'],config['seed'])!=(run_id,count,seed):
            raise ValueError('Full shard intent differs from the frozen input')
        pilot=config['pilot_admission'];receipt=config['publisher_receipt']
        if (local or pilot['full_render_admission']!='PASS' or
                receipt['status']!='DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED' or
                receipt['folder_id']!=folder_id or receipt['secret_cleanup']!='CONFIRMED_DELETED'):
            raise ValueError('Full shard requires accepted Pilot and publisher receipts')
        start_index=config['start_index']
        if (type(start_index) is not int or start_index<0 or start_index+count>config['total_scenes'] or
                (input_root/'SCENE_PLAN.json').exists()):
            raise ValueError('Full shard range differs from the dataset plan')
    gate = json.loads((input_root/'SMOKE_HUMAN_REVIEW.json').read_text(encoding='utf8'))
    if gate['status'] != 'PASS' or gate['reviewed_images'] < 30 or gate['pilot_authorized'] is not True:
        raise ValueError('Smoke human gate has not passed')
    scratch=Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path('/tmp')
    run_root = (scratch if publisher else work_root)/run_id
    run_root.mkdir(parents=True, exist_ok=False)
    result = run_root/'result'
    result.mkdir()
    started = time.monotonic()
    deadline = started+max_seconds
    compute_deadline = deadline-min(600,max(30,max_seconds//10))
    state = {'schema':'topface-native-pilot-v1.1','run_id':run_id,'status':'STARTED',
        'observed_utc':datetime.now(timezone.utc).isoformat(), 'optimizer_steps':0,
        'requested_scenes':count,'seed':seed,'automatic_replay':False,
        'scope':'LOCAL_WORKER_QUALIFICATION' if local else ('KAGGLE_NATIVE_BLENDER_FULL_SHARD' if publisher else 'KAGGLE_NATIVE_BLENDER_PILOT'),
        'start_index':start_index,
        'input_manifest_sha256':sha(input_root/'INPUT_MANIFEST.json'),
        'source_files':manifest['files'],'max_seconds':max_seconds,
        'stopping_condition':'complete requested scenes or finite deadline; no automatic scientific replay'}
    drive = None
    def stage(name):
        state['stage'] = name
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(result/'STATE.json',state)
        print('TOPFACE_STAGE',json.dumps({k:state[k] for k in ('run_id','stage','elapsed_seconds')}),flush=True)
    try:
        stage('ENVIRONMENT_AND_STORAGE_ADMISSION')
        if shutil.disk_usage(work_root).free < 8*1024**3:
            raise RuntimeError('Less than registered 8 GiB free working space')
        if publisher and shutil.disk_usage(scratch).free < 4*1024**3:
            raise RuntimeError('Insufficient cloud scratch space for Blender and render output')
        environment = {'python':sys.version,'local':local,'packages':{},'blenderproc_required':False}
        for package in ('numpy','opencv-python','torch','torchvision'):
            try:environment['packages'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:environment['packages'][package] = None
        if not local:
            import torch
            environment.update(pytorch_gpu_available=torch.cuda.is_available(),
                pytorch_gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            if publisher:
                state['publisher_route_qualification']=receipt['github']
            else:
                drive = Drive(folder_id,deadline=deadline)
                drive.verify_folder()
                atomic_json(result/'admission-probe.json',{'run_id':run_id,'route':'notebook-direct','producer':'Kaggle'})
                state['drive_probe'] = drive.export(result/'admission-probe.json',run_id+'-admission',result/'DRIVE_PROBE.json')
                state['drive_admission_marker'] = drive.reserve_run(run_id,state['input_manifest_sha256'])
        atomic_json(result/'environment.json',environment)
        source = result/'source'
        source.mkdir()
        for member in manifest['files']:
            if member['path'].endswith(('.py','.json','.yaml')):
                shutil.copy2(input_root/member['path'],source/Path(member['path']).name)
        assets = result/'cad'
        extract_cad(input_root,assets)
        stage('NATIVE_BLENDER_RUNTIME')
        blender = Path(local_blender) if local else install_blender(run_root,compute_deadline)
        atomic_json(result/'blender_runtime.json',{'url':BLENDER_URL,'sha256':BLENDER_SHA256,
            'local_binary':str(blender) if local else None,'local_binary_sha256':sha(blender) if local else None})
        stage('ONE_FRAME_RUNTIME_QUALIFICATION')
        try:
            render(blender,input_root,assets,result/'qualification-auto',1,seed,'auto',
                   result/'qualification-auto.log',compute_deadline,qualification=True)
            selected = json.loads((result/'qualification-auto/render_device.json').read_text())
        except (RuntimeError, subprocess.TimeoutExpired):
            # Exactly one CPU fallback, only for the one-frame runtime probe.
            # Full Pilot is not automatically repeated after a partial render.
            state['qualification_cpu_fallback'] = True
            render(blender,input_root,assets,result/'qualification-cpu',1,seed,'cpu',
                   result/'qualification-cpu.log',compute_deadline,qualification=True)
            selected = json.loads((result/'qualification-cpu/render_device.json').read_text())
        state['qualification_device'] = selected
        from audit_v11 import audit
        qualification = result/('qualification-cpu' if state.get('qualification_cpu_fallback') else 'qualification-auto')
        audit(qualification)
        stage('PILOT_RENDER')
        render(blender,input_root,assets,result/'pilot',count,seed,
               'cpu' if selected['selected_backend']=='CPU' else 'auto',
               result/'pilot-render.log',compute_deadline,start_index=start_index)
        stage('PILOT_LABEL_QA')
        report = audit(result/'pilot')
        if report['images'] != count:
            raise RuntimeError('Pilot scene count mismatch')
        if publisher:
            from coco_topface import export
            export(result/'pilot',report)
        state.update(status='COMPUTE_AND_QA_PASS',images_rendered=report['images'],instances=report['instances'],
            qa={k:v for k,v in report.items() if k!='scenes'},
            device=json.loads((result/'pilot/render_device.json').read_text()))
    except Exception as error:
        state.update(status='FAILED_OR_BUDGET_STOP',error_type=type(error).__name__)
        progress = result/'pilot/PROGRESS.json'
        if progress.is_file():
            state['render_progress'] = json.loads(progress.read_text())
        # Preserve the stage and private-free failure class. Raw library
        # exception strings can contain credential endpoints.
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(result/'STATE.json',state)
        compact = work_root/(run_id+'-RESULT.json')
        state['durable_status'] = 'ARCHIVE_PENDING'
        atomic_json(compact,state)
        archive = work_root/(run_id+'.tar.gz')
        try:
            archive_directory(result,archive,deadline=deadline)
            state['archive'] = {'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive,deadline)}
            atomic_json(compact,state)
            if drive is not None:
                state['drive_archive'] = drive.export(archive,run_id,work_root/(run_id+'-DRIVE.json'))
                state['durable_status'] = 'FULL_BYTE_SHA256_PASS'
                if state.get('drive_admission_marker'):
                    try:drive.finish_run(state['drive_admission_marker'],state['status'])
                    except Exception:state['admission_marker_final_status_update']='PENDING_RECONCILIATION'
            else:
                state['durable_status'] = ('LOCAL_ONLY' if local else
                    'PUBLISHER_PENDING_EXACT_SAVED_OUTPUT' if publisher else 'NO_DRIVE_ADMISSION')
        except Exception as error:
            state['durable_status'] = 'UNACCEPTED_PRESERVE_SAVED_OUTPUT'
            state['publication_error_type'] = type(error).__name__
        state['elapsed_seconds'] = time.monotonic()-started
        atomic_json(compact,state)
        print('TOPFACE_RESULT',json.dumps(state),flush=True)
    # A publisher-route failure archive remains available for investigation.
    if not publisher and (state['status'] != 'COMPUTE_AND_QA_PASS' or (not local and state['durable_status']!='FULL_BYTE_SHA256_PASS')):
        raise RuntimeError('Finite native job did not satisfy compute/storage acceptance; inspect compact RESULT')
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--work-root',type=Path,default=Path('/kaggle/working'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261115)
    parser.add_argument('--max-seconds',type=int,default=9000)
    parser.add_argument('--local-blender',type=Path)
    args=parser.parse_args()
    main(**vars(args))
