"""Collect fresh live TOP results for six targets, then keep the preview running."""
import sys,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
sys.path.insert(0,'/sdcard/transport_mvp')
import k230.visual_inspector as module
from k230.capture_io import CaptureIO
from run_block_preview import main
ROOT='/sdcard/transport_mvp/'
PROFILE_FILE=ROOT+'vision_pixel_profile.json'
REPORT_FILE=ROOT+'top_validation_latest.json'
profile=json.load(open(PROFILE_FILE))
Original=module.VisualInspector
storage=CaptureIO()
rows={str(c):{'frames':0,'pixels_found':0,'stable_frames':0,'errors':0,'mean_center_px':[0.0,0.0],
    'min_center_px':None,'max_center_px':None,'last':None} for c in range(1,7)}


class TopCheck(Original):
    def __init__(self,*a,**k):
        self.seen=0;self.current=-1;self.finished=False
        super().__init__(*a,**k)

    def poll_file(self,path):
        if self.finished:return super().poll_file(path)
        if self.current>=0 and rows[str(self.target_id)]['frames']<100:return False
        index=self.current+1
        self.current=index
        command={'seq':300+index,'mode':'TOP','target_id':index+1 if index<6 else 1,'station':'BUFFER'}
        if index==6:self.finished=True
        storage.save_json(command,path)
        return self.apply(command)

    def process(self,frame):
        result=super().process(frame)
        if self.finished:return result
        self.seen+=1
        row=rows[str(self.target_id)];row['frames']+=1
        if result['status']=='ERROR':row['errors']+=1
        if result['u_px'] is not None:
            row['pixels_found']+=1;n=row['pixels_found'];point=[result['u_px'],result['v_px']]
            if row['min_center_px'] is None:row['min_center_px']=list(point);row['max_center_px']=list(point)
            for axis in (0,1):
                row['mean_center_px'][axis]+=(point[axis]-row['mean_center_px'][axis])/n
                row['min_center_px'][axis]=min(row['min_center_px'][axis],point[axis])
                row['max_center_px'][axis]=max(row['max_center_px'][axis],point[axis])
        row['stable_frames']+=bool(result['stable']);row['last']=result
        if result['x_mm'] is not None or result['target_pose_valid']:raise ValueError('Uncalibrated TOP produced a metric motion pose')
        if row['frames']==100:
            report={'test':'REAL_CSI1_TOP_CANDIDATE_FRESH_FRAMES','source':'REAL_CAMERA','camera_id':1,
                'profile':profile,'per_color':rows,'processed_frames':self.seen,
                'scope':'Sequential same-scene pixel availability; no independent accuracy or millimeter claim',
                'complete':all(r['frames']>=100 for r in rows.values()),'motors_enabled':False,
                'storage':storage.status()}
            storage.save_json(report,REPORT_FILE)
            print('TOP_LIVE_STAGE:'+json.dumps({'frames':self.seen,'target':self.target_id,'row':row}))
        return result


module.VisualInspector=TopCheck
try:
    with open(ROOT+'confirmed_task_receipt.json') as stream:task=json.load(stream)
except OSError:task=None
try:
    main(inspection_mode='TOP',inspection_target=1,inspection_station='BUFFER',pixel_profile=profile,
        control_path=ROOT+'vision_control.json',confirmed_task=task,report_path=ROOT+'top_live_latest.json')
finally:
    module.VisualInspector=Original
