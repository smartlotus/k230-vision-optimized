"""Whole task through nonblocking intents and delayed SIM controller receipts."""
import argparse
import json
from pathlib import Path
from core.workflow_session import WorkflowSession
from core.task_parser import parse_task
from core.types import Step
from sim.field_sim import FieldSim
from sim.motion_sim import MotionSim
from sim.camera_sim import CameraSim
from sim.run_mission import FAULTS


class AsyncSimController:
    """ACK reserves an intent; the next poll executes the independent field model."""
    def __init__(self,motion):
        self.motion=motion
        self.pending=None
        self.ack_sent=False
        self.seq=0
        self.submissions=0
        self.receipts=[]

    def submit(self,intent):
        if self.pending is not None:raise ValueError('One outstanding controller intent')
        if intent['is_simulated'] is not True:raise ValueError('SIM controller refuses LIVE intent')
        self.pending=intent;self.ack_sent=False;self.submissions+=1

    def poll(self):
        if self.pending is None:return None
        intent=self.pending;step=intent['step'];self.seq+=1
        receipt={'session_id':intent['session_id'],'request_id':intent['request_id'],
                 'seq':self.seq,'is_simulated':True,'status':'ACK'}
        if not self.ack_sent:
            self.ack_sent=True
        else:
            obj=Step(**step)
            self.motion.prepare(obj,intent['observation'])
            action=step['action']
            if action=='MOVE':result=self.motion.drive_to(step['dest_station'])
            elif action=='PICK':result=self.motion.pick_to_carrier(step['target_color'],step['source_station'],step['target_slot'])
            elif action=='PLACE':result=self.motion.place_from_carrier(step['target_color'],step['dest_station'],step['target_slot'],step['layer'])
            else:result=self.motion.return_to_start(self.motion.field.start_id)
            receipt.update(status=result['status'],reason=result.get('reason'),action=action,
                batch=step['batch'],color_id=step['target_color'],slot=step['target_slot'],layer=step['layer'],
                station=self.motion.field.station,carrier_count=len(self.motion.field.carrier),
                gripper_empty=self.motion.field.gripper is None,
                loaded_verified=action=='PICK' and result['status']=='DONE',
                placement_verified=action=='PLACE' and result['status']=='DONE',
                simulation_command_id=result['command_id'])
            self.pending=None
        self.receipts.append(dict(receipt))
        return receipt


def run_workflow(payload='156+123+516+231',capacity=3,start_id='START_1',inject=None):
    if inject is not None and inject not in FAULTS:raise ValueError('Unknown fault')
    field=FieldSim(parse_task(payload),capacity,start_id)
    # Explicit simulated QR-site arrival; not a real navigation assertion.
    field.drive('QR')
    motion=MotionSim(field,inject)
    controller=AsyncSimController(motion)
    session=WorkflowSession(payload,'sim-workflow',mode='SIM',capacity=capacity,start_id=start_id)
    camera=CameraSim(payload,inject)
    now=0
    for _ in range(256):
        if session.status in ('DONE','ERROR','UNCERTAIN'):break
        req=session.requirement()
        observation=None
        if req['step']['action'] in ('PICK','PLACE'):
            frame=camera.get_frame(req['camera_id'])
            observation=camera.detect(req['kind'],frame,req['target_id'],req['station'])
            now=frame['timestamp_ms']
        result=session.issue(observation,now_ms=now)
        if result['intent'] is None:break
        controller.submit(result['intent'])
        before=session.step_index
        session.receive(controller.poll(),now)
        if session.step_index!=before:raise ValueError('ACK incorrectly completed a step')
        session.receive(controller.poll(),now+20)
        now+=40
    return {'source':'NONBLOCKING_DETERMINISTIC_SIM_ONLY','summary':session.summary(),
            'field_complete':field.is_complete(),'field':field.snapshot(),'transitions':field.transitions,
            'controller_submissions':controller.submissions,'controller_receipts':controller.receipts,
            'journal':session.journal,'events':session.events,'workflow_snapshot':session.snapshot(),
            'injection':inject,'max_carrier_occupancy':motion.max_carrier_occupancy}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='156+123+516+231')
    parser.add_argument('--capacity',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--start',choices=('START_1','START_2'),default='START_1')
    parser.add_argument('--inject',choices=FAULTS)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=run_workflow(args.task,args.capacity,args.start,args.inject)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'summary':report['summary'],'field_complete':report['field_complete']}))
    raise SystemExit(0 if report['summary']['status']=='DONE' and report['field_complete'] else 1)
