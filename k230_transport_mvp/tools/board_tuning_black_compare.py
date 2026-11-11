"""Same-fresh-frame black LAB comparison; no auto-save or motion."""
import sys,os,time,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
ROOT='/sdcard/transport_mvp/'
REPORT=ROOT+'top640_black_angle02_pair100.json'
try:os.stat(REPORT)
except OSError:pass
else:raise RuntimeError('Existing result must not be overwritten')
sys.path.insert(0,ROOT)
from k230.tuning.runtime import TuningApp
from k230.tuning.schema import clone,compile_config
from k230.vision_adapter import VisionAdapter
from k230.capture_io import CaptureIO
p=json.load(open(ROOT+'top640_angle01_candidate.json'));q=clone(p)
q['profile_id']='csi1-top640-angle02-black17';q['revision']=2
q['contexts']['TOP_BUFFER']['thresholds']['5'][0]=17
configs={'original':p,'L17':q};adapters={k:VisionAdapter(compile_config(v)) for k,v in configs.items()}
app=TuningApp();app.session.draft=q;app.session.dirty=True;app.session.apply();app.session.select('TOP','BUFFER');app.session.target_id=5
rows=[];last=-1;start=None;done=False
summary={k:{'frames':0,'found':0,'stable':0,'within_face_12px':0,'errors':0} for k in configs}
guard=(clone(app.session.active),clone(app.session.draft),app.session.slot,app.session.apply_serial)
try:
 app.setup();app.panel.live_requested=True;started=time.ticks_ms()
 while True:
  app.tick()
  if not done and (app.session.active,app.session.draft,app.session.slot,app.session.apply_serial)!=guard:raise RuntimeError('Operator changed tested profile')
  if done or app.frame is None or app.frame_id==last or time.ticks_diff(time.ticks_ms(),started)<2500:continue
  last=app.frame_id
  if start is None:start=time.ticks_ms()
  frame=app.frame.copy();packet=dict(app.frame_metadata);packet['image']=frame
  results={};raw={}
  for name,adapter in adapters.items():
   obs=adapter.detect('TOP',packet,5,'BUFFER');results[name]=obs;r=summary[name]
   r['frames']+=1;r['found']+=obs['u_px'] is not None;r['stable']+=bool(obs['stable']);r['errors']+=obs['status']=='ERROR'
   if obs['u_px'] is not None:r['within_face_12px']+=abs(obs['u_px']-424)<=12 and abs(obs['v_px']-146)<=12
   if len(rows) in (0,99):
    threshold=configs[name]['contexts']['TOP_BUFFER']['thresholds']['5']
    raw[name]=[{'rect':b.rect(),'pixels':b.pixels(),'density':b.pixels()/(b.w()*b.h())} for b in frame.find_blobs([threshold],roi=(170,75,290,205),area_threshold=200,pixels_threshold=20,merge=False)]
  rows.append({'frame_id':last,'results':results,'raw_components':raw})
  if len(rows)>=100:
   result={'test':'REAL_CSI1_TOP640_ANGLE02_BLACK_PAIRED100','source':'REAL_CAMERA','camera_id':1,'is_simulated':False,
    'profiles':configs,'summary':summary,'rows':rows,'frame_denominator':100,'elapsed_ms':time.ticks_diff(time.ticks_ms(),start),
    'camera_errors':app.capture_errors,'saved_generation':None,'motors_enabled':False,
    'limit':'Two adapters on same frozen native RGB565 frame; photo center424,146 +/-12px coarse only. No independent accuracy or H. No auto-save.'}
   temporary=REPORT+'.tmp'
   try:os.stat(temporary)
   except OSError:pass
   else:raise OSError('Existing partial report preserved')
   persisted=CaptureIO().save_json(result,temporary)
   if persisted:
    with open(temporary) as stream:
     if stream.read()!=json.dumps(result):raise OSError('Report readback differs')
    os.sync();os.rename(temporary,REPORT);os.sync()
    with open(REPORT) as stream:persisted=stream.read()==json.dumps(result)
   print('BLACK_PAIR_RESULT:'+json.dumps({'summary':summary,'report_persisted':persisted}));done=True
except BaseException as error:print('BLACK_PAIR_EXIT:'+repr(error));raise
finally:app.close()
