"""Fresh-frame current-scene TOP check, using the actual standalone panel path."""
import sys,os,time,json,math
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
ROOT='/sdcard/transport_mvp/'
sys.path.insert(0,ROOT)
from k230.tuning.runtime import TuningApp
from k230.tuning.schema import clone
from k230.capture_io import CaptureIO
storage=CaptureIO();app=TuningApp();rows={};current=1;done=False;collecting=False
REPORT=ROOT+'top640_angle02_fresh600_attempt01.json'
try:
    os.stat(REPORT)
except OSError:pass
else:raise RuntimeError('Existing test result must not be overwritten')
candidate=json.load(open(ROOT+'top640_angle02_candidate.json'))
app.session.draft=candidate;app.session.dirty=True;app.session.apply()
expected={1:(255,112),2:(333,183),3:(245,202),4:(343,96),5:(424,146),6:(396,248)}
started=None;last_frame=-1;check_started=None
guard=(clone(app.session.active),clone(app.session.draft),app.session.slot,app.session.apply_serial)

def save_report(result,path):
    raw=json.dumps(result);temporary=path+'.tmp'
    try:
        os.stat(temporary)
    except OSError:pass
    else:raise OSError('Existing partial report preserved')
    if not storage.save_json(result,temporary):return False
    with open(temporary) as stream:
        if stream.read()!=raw:raise OSError('Report readback differs')
    os.sync();os.rename(temporary,path);os.sync()
    with open(path) as stream:return stream.read()==raw

try:
    app.setup();app.panel.live_requested=True
    started=time.ticks_ms()
    while True:
        app.tick();now=time.ticks_ms()
        if not done and (app.session.active,app.session.draft,app.session.slot,app.session.apply_serial)!=guard:
            raise RuntimeError('Operator changed tested profile; no automatic save')
        if done:continue
        if not collecting and time.ticks_diff(now,started)>=2500:
            collecting=True;check_started=now;app.session.select('TOP','BUFFER');app.session.target_id=1
            app.session.adapter.reset_session();last_frame=app.frame_id
        if not collecting:continue
        timeout=time.ticks_diff(now,check_started)>=90000
        if app.frame_id==last_frame and not timeout:continue
        if not timeout:
            last_frame=app.frame_id;obs=app.session.observation
            if obs is None:continue
            if obs.get('kind')!='TOP' or obs.get('target_id')!=current or obs.get('camera_id')!=1:raise RuntimeError('Test context changed')
            row=rows.setdefault(str(current),{'frames':0,'found':0,'stable':0,'detector_errors':0,
                'inside_manual_face_region':0,'centers':[],'first_frame':app.frame_id,'last_frame':app.frame_id,
                'manual_photo_center_reference':expected[current],'examples':[]})
            row['frames']+=1;row['last_frame']=app.frame_id
            row['detector_errors']+=obs['status']=='ERROR'
            if obs['u_px'] is not None:
                row['found']+=1;row['stable']+=bool(obs['stable'])
                u,v=obs['u_px'],obs['v_px'];row['centers'].append([u,v])
                row['inside_manual_face_region']+=abs(u-expected[current][0])<=12 and abs(v-expected[current][1])<=12
            if row['frames'] in (1,100):row['examples'].append(clone(obs))
            if row['frames']>=100:
                print('TOP640_COLOR_CHECK:'+json.dumps({'color':current,'frames':row['frames'],
                    'found':row['found'],'stable':row['stable'],'inside_manual_face_region':row['inside_manual_face_region']}))
                current+=1
                if current<=6:
                    app.session.target_id=current;app.session.adapter.reset_session();app.session.observation=None
        if current>6 or timeout:
            complete=current>6 and not timeout
            passed=complete and not app.errors and not app.capture_errors and all(r['found']==100 and r['stable']>=95
                and r['detector_errors']==0 and r['inside_manual_face_region']==r['found'] for r in rows.values())
            committed=None  # Report only; saving is an explicit operator action after review.
            result={'test':'REAL_CSI1_TOP640_ANGLE02_100_PER_COLOR','source':'REAL_CAMERA','camera_id':1,
                'is_simulated':False,'rows':rows,'complete':complete,'passed_current_scene_gate':passed,
                'frame_denominator':sum(r['frames'] for r in rows.values()),
                'elapsed_ms':time.ticks_diff(now,check_started),'processing_fps':sum(r['frames'] for r in rows.values())*1000/max(1,time.ticks_diff(now,check_started)),
                'profile':candidate,'camera_errors':app.capture_errors,'saved_generation':committed['generation'] if committed else None,
                'saved_sha256':committed['sha256'] if committed else None,'motors_enabled':False,
                'manual_region_half_width_px':12,
                'limit':'Same bench pose/light only. Manual12px photo regions are coarse sanity checks, not independent positioning accuracy or H. Prior attempt01 used25px and allowed black body-center hits. Per-color sequential; no simultaneous-six-color rate claim.'}
            persisted=save_report(result,REPORT)
            print('TOP640_CHECK_RESULT:'+json.dumps({'complete':complete,'passed_current_scene_gate':passed,'frames':result['frame_denominator'],'fps':result['processing_fps'],'saved_generation':result['saved_generation'],'report_persisted':persisted}))
            app.session.target_id=5;app.session.adapter.reset_session();app.session.observation=None
            app.panel.page='PREVIEW';app.panel.message=('当前场景检查通过，参数未保存' if passed else '新角度仍需调整，候选未自动保存') if persisted else '报告写入失败，不能声明检查完成'
            done=True
except BaseException as error:
    print('TOP640_PREVIEW_EXIT:'+repr(error))
    raise
finally:
    app.close()
