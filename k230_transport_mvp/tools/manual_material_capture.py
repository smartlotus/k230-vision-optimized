"""One operator-invoked original photo via the existing official CanMV session.

No continuous sampling, second serial owner, camera restart or training. Unknown
sample outcomes are not replayed. The ignored local preview controller must exist.
"""
import argparse,hashlib,json,re,runpy,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,help='Operator scene tag, e.g. G001; retain within one fixed setup')
    p.add_argument('--output',type=Path,default=Path.home()/'Desktop/K230_material_photos')
    p.add_argument('--camera-height-cm',type=float);p.add_argument('--lighting',default='UNMEASURED')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,24}',a.scene):p.error('Scene tag must use1..24letters/digits/underscore/hyphen')
    if a.camera_height_cm is not None and not 0<a.camera_height_cm<1000:p.error('Measured positive height below1000cm required')
    controller=ROOT/'.ops/local/preview_control.py';state_path=ROOT/'.ops/local/canmv-active-resume.json'
    if not controller.exists() or not state_path.exists():p.error('Current live preview controller/state missing; attach the official session first')
    h=runpy.run_path(str(ROOT/'k230-workbench/scripts/check-canmv-host.py'))
    url,headers=h['mcp_connection'](Path.home()/'.codex/config.toml','canmv-k230');m=h['OfficialMcp'](url,headers)
    m.request('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'operator-single-photo','version':'1'}})
    if m.tool('canmv_script_running').get('running') is not True:raise SystemExit('Existing preview is not running; no automatic start/interrupt')
    state=json.loads(state_path.read_text())
    if state.get('pending'):raise SystemExit('Pending outcomeUNKNOWN; do not repeat the sample command')
    lock=ROOT/'.ops/local/manual-material-photo.lock'
    try:stream=lock.open('x',encoding='utf-8')
    except FileExistsError:raise SystemExit('A manual capture lock exists; reconcile that capture before retrying')
    with stream:stream.write('Operator-owned single-photo transaction\n')
    remote='/sdcard/transport_mvp/ellipse_perf_samples';record=None;receipt_path=None
    request_attempted=False;saved_complete=False
    try:
        entries=m.tool('canmv_list_dir',{'path':remote})['entries']
        serials={int(e['name'][7:13]) for e in entries if e['name'].startswith('sample-') and e['name'][7:13].isdigit()}
        serial=max(serials or {0})+1;prefix='sample-%06d'%serial
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');stem=a.scene+'-'+stamp+'-%06d'%serial
        output=a.output.resolve()
        for folder in ('images','raw','captures'):(output/folder).mkdir(parents=True,exist_ok=True)
        receipt_path=output/'captures'/(stem+'.capture.json')
        if receipt_path.exists():raise ValueError('Capture receipt already exists')
        record={'status':'REQUEST_PREPARED','operator_scene_tag':a.scene,'scene_group_confirmed':False,
            'camera_height_cm':a.camera_height_cm,'lighting':a.lighting,'labels_status':'UNREVIEWED',
            'human_approved':False,'training_started':False,'source_session_sha256':hashlib.sha256(state['token'].encode()).hexdigest()}
        record.update(expected_serial=serial,expected_command_seq=state.get('seq',0)+1)
        receipt_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        lock.write_text(json.dumps({'receipt_path':str(receipt_path),'expected_serial':serial,
             'expected_command_seq':record['expected_command_seq'],'status':'REQUEST_IN_PROGRESS'})+'\n',encoding='utf-8')
        request_attempted=True
        run=subprocess.run([sys.executable,'-B','-X','utf8',str(controller),'sample'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        if run.returncode:raise ValueError('Sample request failed orUNKNOWN; no replay')
        ack=json.loads(run.stdout.strip().splitlines()[-1])
        if ack.get('result')!='ACK' or ack.get('ok') is not True:raise ValueError('No verified sample ACK; do not repeat')
        if ack.get('seq')!=record['expected_command_seq']:raise ValueError('Another command changed the expected sample sequence')
        deadline=time.monotonic()+25;complete=False
        while time.monotonic()<deadline:
            entries=m.tool('canmv_list_dir',{'path':remote})['entries'];files={e['name']:e for e in entries}
            current={int(e['name'][7:13]) for e in entries if e['name'].startswith('sample-') and e['name'][7:13].isdigit()}
            if current-serials not in (set(),{serial}):raise ValueError('Concurrent capture changed the serial; no automatic reassignment')
            if (files.get(prefix+'-native.rgb565',{}).get('size')==460800
                    and files.get(prefix+'-lab-grid.json',{}).get('size',0)>1000
                    and files.get(prefix+'-live.jpg',{}).get('size',0)>1000):complete=True;break
            if m.tool('canmv_script_running').get('running') is not True:raise ValueError('Preview stopped during capture')
            time.sleep(.5)
        if not complete:raise ValueError('Sample persistenceUNKNOWN; do not request a replacement')
        saved={}
        for suffix,extension in (('-live.jpg','.jpg'),('-lab-grid.json','-lab.json'),('-native.rgb565','.rgb565')):
            target=output/'raw'/(stem+extension)
            m.tool('canmv_download_file_to_host',{'path':remote+'/'+prefix+suffix,'localPath':str(target)})
            saved[extension]={'path':target.relative_to(output).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        meta=json.loads((output/saved['-lab.json']['path']).read_text())
        if (meta.get('camera_id')!=2 or meta.get('is_simulated') is not False or meta.get('source')!='REAL_CAMERA_NATIVE_LAB'
                or (meta.get('width'),meta.get('height'))!=(640,360)):
            raise ValueError('Unexpected camera sample identity')
        if json.loads(state_path.read_text())['token']!=state['token']:raise ValueError('Live session changed during capture')
        import cv2,numpy as np
        native=(output/saved['.rgb565']['path']).read_bytes()
        if len(native)!=460800:raise ValueError('Incomplete native source')
        bgr=cv2.cvtColor(np.frombuffer(native,np.uint8).reshape(360,640,2),cv2.COLOR_BGR5652BGR)
        image_path=output/'images'/(stem+'.png')
        if not cv2.imwrite(str(image_path),bgr):raise OSError('Original image export failed')
        record.update(status='SAVED_UNREVIEWED',frame_id=meta['frame_id'],timestamp_ms=meta['timestamp_ms'],
            camera_id=2,source_wh=[640,360],command_seq=ack['seq'],files=saved,
            image_file=image_path.relative_to(output).as_posix(),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
        receipt_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        saved_complete=True
        print(json.dumps({'status':record['status'],'original_image':str(image_path),'receipt':str(receipt_path),
                          'training_started':False},ensure_ascii=False))
    except Exception as error:
        if record is not None and receipt_path is not None:
            record.update(status='INCOMPLETE_OR_UNKNOWN',error=str(error))
            receipt_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if request_attempted:
            lock.write_text(json.dumps({'status':'UNKNOWN_RECONCILE_BEFORE_NEW_SAMPLE','receipt_path':str(receipt_path)})+'\n',encoding='utf-8')
        raise
    finally:
        if saved_complete or not request_attempted:lock.unlink()


if __name__=='__main__':main()
