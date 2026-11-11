"""Bounded native five-mode switching check; existing scene, no target-placement claim."""
import sys
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
sys.path.insert(0,'/sdcard/transport_mvp')
import json
import k230.visual_inspector as inspector_module
from run_block_preview import main

ROOT='/sdcard/transport_mvp'
CONTROL=ROOT+'/vision_control.json'
Original=inspector_module.VisualInspector
samples=[]
SCHEDULE=((0,'COLOR',1,'RAW'),(120,'TOP',1,'BUFFER'),(200,'RING',1,'PROCESS'),
          (260,'LANE',None,None),(320,'QR',None,'QR'),(400,'IDLE',None,None),
          (460,'COLOR',1,'RAW'))


class ScheduledInspector(Original):
    def __init__(self,*args,**kwargs):
        self.seen=0;self.schedule_index=-1;self.sample_modes=set()
        Original.__init__(self,*args,**kwargs)

    def poll_file(self,path):
        index=0
        for i,item in enumerate(SCHEDULE):
            if self.seen>=item[0]:index=i
        if index!=self.schedule_index:
            _,mode,target,station=SCHEDULE[index]
            with open(path,'w') as stream:
                stream.write(json.dumps({'seq':100+index,'mode':mode,'target_id':target,'station':station}))
            self.schedule_index=index
        return Original.poll_file(self,path)

    def process(self,frame):
        self.seen+=1
        result=Original.process(self,frame)
        if self.mode not in self.sample_modes:
            self.sample_modes.add(self.mode)
            samples.append({'state':self.status(),'observation':result})
        if result is not None:
            if result['x_mm'] is not None or result['target_pose_valid']:
                raise ValueError('Uncalibrated inspection unexpectedly produced a motion pose')
            if result['camera_id']!=self.camera_id():raise ValueError('Wrong live source camera')
        return result


inspector_module.VisualInspector=ScheduledInspector
try:
    with open(ROOT+'/confirmed_task_receipt.json') as stream:task=json.load(stream)
except OSError:task=None
try:
    report=main(max_frames=600,inspection_mode='COLOR',inspection_target=1,inspection_station='RAW',
        confirmed_task=task,control_path=CONTROL,report_path=ROOT+'/vision_modes_attempt02.json')
finally:
    inspector_module.VisualInspector=Original
report['test']='LIVE_FIVE_MODE_CONTROL_CHECK'
report['mode_entry_samples']=samples
report['all_modes_exercised']=set(report['per_mode_frames'])==set(Original.MODES)
report['scope']='Real camera acquisition and mode controls; no live TOP/RING/LANE target acceptance'
report['check_passed']=report['fatal'] is None and report['all_modes_exercised'] and not any(report['errors'].values())
with open(ROOT+'/vision_modes_attempt02.json','w') as stream:stream.write(json.dumps(report))
print('VISION_CHECK:'+json.dumps({'check_passed':report['check_passed'],'frames':report['frames'],
    'per_mode_frames':report['per_mode_frames'],'per_mode_status':report['per_mode_status'],
    'errors':report['errors'],'mode_entry_samples':samples}))
