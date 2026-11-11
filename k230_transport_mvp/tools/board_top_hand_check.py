"""Operator-labelled hand/no-hand phases with actual frames; preview stays alive."""
import sys,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
sys.path.insert(0,'/sdcard/transport_mvp')
import k230.visual_inspector as module
from k230.capture_io import CaptureIO
from run_block_preview import main
ROOT='/sdcard/transport_mvp/'
profile=json.load(open(ROOT+'top_hand_candidate.json'))
baseline_profile=json.load(open(ROOT+'vision_pixel_profile.json'))
Original=module.VisualInspector
storage=CaptureIO();rows={}


class HandCheck(Original):
    def __init__(self,*args,**kwargs):
        self.phase=None
        super().__init__(*args,**kwargs)
        self.baseline=Original('TOP',5,'BUFFER',baseline_profile)

    def process(self,frame):
        phase=None
        try:
            with open(ROOT+'hand_phase.json') as stream:request=json.loads(stream.read(256))
            if request.get('phase') in ('NO_HAND','HAND_TOP_VISIBLE','TOP_OCCLUDED'):phase=request['phase']
        except (OSError,ValueError,TypeError):pass
        if phase!=self.phase:
            self.capture_failed();self.baseline.capture_failed();self.phase=phase
        result=super().process(frame)
        if phase is None or rows.get(phase,{}).get('frames',0)>=100:return result
        baseline=self.baseline.process(frame)
        row=rows.setdefault(phase,{'frames':0,'found':0,'stable':0,'errors':0,'cleared':0,
            'baseline_found':0,'baseline_stable':0,
            'first_frame':None,'last_frame':None,'examples':[],'phase_source':'OPERATOR_LABEL_NOT_AUTOMATIC_HAND_DETECTOR'})
        if row['frames']>=100:return result
        row['frames']+=1
        row['first_frame']=row['first_frame'] or frame['frame_id'];row['last_frame']=frame['frame_id']
        row['found']+=result['u_px'] is not None;row['stable']+=bool(result['stable']);row['errors']+=result['status']=='ERROR'
        row['baseline_found']+=baseline['u_px'] is not None;row['baseline_stable']+=bool(baseline['stable'])
        row['cleared']+=result['u_px'] is None and result['v_px'] is None and not result['stable']
        if row['frames'] in (1,50,100):
            path=ROOT+'top-hand-boundary06-'+phase+'-'+str(row['frames'])+'-live.jpg'
            saved=storage.save_image(frame['image'],path)
            row['examples'].append({'observation':result,'baseline':baseline,'path':path,'saved':saved,
                'circle_diagnostics_first40':self.adapter.last_top_circles})
        if row['frames']==100:
            report={'test':'REAL_CSI1_TOP_HAND_PHASES','source':'REAL_CAMERA','profile':profile,'phases':rows,
                'motors_enabled':False,'storage':storage.status(),
                'scope':'Operator-labelled 100-frame phases at current bench; no automatic human detector or movement authority'}
            storage.save_json(report,ROOT+'top_hand_boundary06.json')
            print('TOP_HAND_PHASE:'+json.dumps({'phase':phase,'row':row}))
        return result


module.VisualInspector=HandCheck
try:
    with open(ROOT+'confirmed_task_receipt.json') as stream:task=json.load(stream)
except OSError:task=None
try:
    main(inspection_mode='TOP',inspection_target=5,inspection_station='BUFFER',pixel_profile=profile,
        control_path=ROOT+'vision_control.json',confirmed_task=task,report_path=ROOT+'top_hand_latest.json')
finally:
    module.VisualInspector=Original
