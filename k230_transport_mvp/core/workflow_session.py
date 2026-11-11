"""Nonblocking task/observation/receipt ledger. This module sends no motor commands."""
import json
from core.task_parser import parse_task
from core.task_planner import build_execution_steps
from k230.camera_roles import MODE_CAMERA
from ports.motion_port import SafetyGate, monotonic_ms, _elapsed
from vision.placement_score import score_placement_receipt


def clone(value):
    return json.loads(json.dumps(value))


class WorkflowSession:
    def __init__(self, payload, session_id, mode='LIVE', capacity=3, start_id='START_1',
                 controller_verified=None, receipt_timeout_ms=5000, clock_ms=None):
        if mode not in ('SIM','LIVE'):
            raise ValueError('Explicit SIM or LIVE mode required')
        if not isinstance(session_id,str) or not 1<=len(session_id)<=64:
            raise ValueError('A unique session ID is required')
        if start_id not in ('START_1','START_2'):
            raise ValueError('Unknown start station')
        if type(receipt_timeout_ms) is not int or not 1<=receipt_timeout_ms<=60000:
            raise ValueError('Invalid receipt timeout')
        self.task=parse_task(payload)
        self._steps=tuple(clone(s.as_dict()) for s in build_execution_steps(self.task,capacity))
        self.session_id,self.mode,self.capacity,self.start_id=session_id,mode,capacity,start_id
        self.is_simulated=mode=='SIM'
        self.controller_verified=self.is_simulated if controller_verified is None else controller_verified is True
        self.timeout_ms=receipt_timeout_ms
        self.clock_ms=clock_ms or monotonic_ms
        self.status='READY' if self.is_simulated else 'WAIT_POSITION'
        self.reason=None
        self.station='QR' if self.is_simulated else None
        self.station_verified=self.is_simulated
        self.step_index=0
        self.counter=0
        self.last_receipt_seq=-1
        self.pending=None
        self.stopped=False
        self.failed=False
        self.failure_receipt=None
        self.carrier=[]
        self.objects={(b,c):'RAW' for b,colors in ((1,self.task.batch1_colors),(2,self.task.batch2_colors)) for c in colors}
        self.process={}
        self.buffer={}
        self.completed={key:0 for key in ('1_PROCESS','1_BUFFER','2_PROCESS','2_BUFFER')}
        self.picks=self.placements=0
        self.consumed_frames={}
        self.site_receipt=None
        self.journal=[]
        self.events=[]
        self.rejected_receipts=0

    def _event(self,kind,**values):
        entry={'event':kind,'session_id':self.session_id,'is_simulated':self.is_simulated,
               'step_index':self.step_index}
        entry.update(values)
        self.events.append(entry)
        self.events=self.events[-128:]

    def _receipt_source(self,receipt,request_id):
        if not self.controller_verified:
            return 'CONTROLLER_UNVERIFIED'
        if not isinstance(receipt,dict):return 'INVALID_RECEIPT'
        if receipt.get('session_id')!=self.session_id:return 'RECEIPT_SESSION_MISMATCH'
        if receipt.get('request_id')!=request_id:return 'RECEIPT_REQUEST_MISMATCH'
        if receipt.get('is_simulated') is not self.is_simulated:return 'RECEIPT_SOURCE_MISMATCH'
        seq=receipt.get('seq')
        if type(seq) is not int or not 0<=seq<=4294967295 or seq<=self.last_receipt_seq:
            return 'STALE_RECEIPT'
        return None

    def verify_position(self,receipt):
        if self.failed:return self._reject('FAILED_SESSION_REQUIRES_AUDIT')
        if self.pending is not None:return self._reject('PENDING_REQUIRES_RECONCILIATION')
        expected=self.station or 'QR'
        error=self._receipt_source(receipt,self.session_id+':POSITION')
        if error:return self._reject(error)
        if (receipt.get('status')!='DONE' or receipt.get('action')!='POSITION'
                or receipt.get('station')!=expected or receipt.get('start_id')!=self.start_id
                or receipt.get('gripper_empty') is not True):
            return self._reject('POSITION_PROOF_REQUIRED')
        self.station=expected
        self.station_verified=True
        self.last_receipt_seq=receipt['seq']
        if self.site_receipt is None and self.step_index==0:self.site_receipt=clone(receipt)
        self.status='STOPPED' if self.stopped else 'READY'
        self.reason=None
        self._event('POSITION_VERIFIED',station=expected)
        return self.summary()

    def requirement(self):
        if self.step_index>=len(self._steps):return {'kind':'DONE','step':None}
        if not self.station_verified:
            return {'kind':'POSITION','camera_id':None,'station':self.station or 'QR',
                    'request_id':self.session_id+':POSITION','step':clone(self._steps[self.step_index])}
        step=clone(self._steps[self.step_index])
        action=step['action']
        kind='COLOR' if action=='PICK' else ('TOP' if action=='PLACE' and step['layer']==2 else 'RING')
        if action in ('MOVE','RETURN'):kind='LANE'
        target=step['target_color'] if kind in ('COLOR','TOP') else step['target_slot'] if kind=='RING' else None
        station=step['source_station'] if action=='PICK' else step['dest_station'] if action=='PLACE' else self.station
        return {'kind':kind,'camera_id':MODE_CAMERA[kind],'target_id':target,
                'station':station,'step':step,'step_index':self.step_index}

    def _precondition(self,step):
        b,c=step['batch'],step['target_color']
        action=step['action']
        key=(b,c)
        if b==2 and self.completed['1_BUFFER']!=3:return 'FIRST_BATCH_INCOMPLETE'
        if 'CARRIER_EMPTY' in step['preconditions'] and self.carrier:return 'CARRIER_NOT_EMPTY'
        if 'CARRIER_LOADED' in step['preconditions'] and not self.carrier:return 'CARRIER_NOT_LOADED'
        if action in ('MOVE','RETURN') and self.station!=step['source_station']:return 'WRONG_SOURCE_STATION'
        if action=='PICK':
            source=step['source_station']
            if self.station!=source or self.objects.get(key)!=source:return 'OBJECT_NOT_AT_SOURCE'
            if len(self.carrier)>=self.capacity:return 'CARRIER_FULL'
            if source=='PROCESS':
                if self.completed[str(b)+'_PROCESS']!=3:return 'PROCESS_PHASE_INCOMPLETE'
                if self.process.get(step['target_slot'])!=key:return 'PROCESS_SLOT_MISMATCH'
        if action=='PLACE':
            dest=step['dest_station'];slot=step['target_slot']
            if self.station!=dest or key not in self.carrier:return 'CARGO_OR_STATION_MISMATCH'
            if dest=='PROCESS' and slot in self.process:return 'PROCESS_SLOT_OCCUPIED'
            if dest=='BUFFER':
                layers=self.buffer.get(slot,[])
                if len(layers)!=step['layer']-1:return 'BUFFER_LAYER_MISMATCH'
                if step['layer']==2 and layers[0]!=(1,c):return 'STACK_COLOR_MISMATCH'
                if self.task.color_to_buffer_slot[c]!=slot:return 'BUFFER_COLOR_SLOT_MISMATCH'
        if action=='RETURN' and (self.carrier or self.completed['2_BUFFER']!=3):return 'RETURN_PRECONDITION'
        return None

    def _observation_error(self,observation,context,now,historical=False):
        req=self.requirement();step=req['step']
        if self.is_simulated and step['action'] in ('MOVE','RETURN'):return None
        obs=observation or {}
        if (obs.get('kind')!=req['kind'] or obs.get('target_id')!=req['target_id']
                or obs.get('station')!=req['station'] or obs.get('camera_id')!=req['camera_id']):
            return 'OBSERVATION_REQUIREMENT_MISMATCH'
        if obs.get('is_simulated') is not self.is_simulated:return 'OBSERVATION_SOURCE_MISMATCH'
        if obs.get('status')!='OK' or obs.get('stable') is not True:return 'TARGET_NOT_READY'
        if type(obs.get('frame_id')) is not int or obs['frame_id']<1:return 'FRAME_ID_UNKNOWN'
        if obs['frame_id']<=self.consumed_frames.get(str(obs['camera_id']),-1):return 'OBSERVATION_ALREADY_CONSUMED'
        age=obs.get('stale_after_ms',250)
        if (type(obs.get('timestamp_ms')) is not int or type(age) is not int or not 1<=age<=5000
                or not 0<=_elapsed(now,obs['timestamp_ms'])<=age):return 'OBSERVATION_STALE'
        if self.is_simulated:
            if obs.get('coord_frame')!='ROBOT_MM' or obs.get('target_pose_valid') is not True:
                return 'SIM_POSE_REQUIRED'
            return None
        if historical:
            # Validate saved source/freshness without republishing historical authorization.
            return None
        ctx=dict(context or {})
        ctx['required_target_id']=req['target_id']
        if step['action'] in ('MOVE','RETURN'):
            destination=self.start_id if step['action']=='RETURN' else step['dest_station']
            allowed,reason=SafetyGate().check_navigation(obs,now,ctx,destination)
        else:
            allowed,reason=SafetyGate().check(obs,now,ctx)
        return None if allowed else reason

    def issue(self,observation=None,context=None,now_ms=None):
        """Reserve one authorized intent BEFORE caller hands it to a controller.

        This returns data only. No GPIO/UART/transport or motion function is called.
        """
        if self.pending is not None:return {'status':self.status,'reason':'REQUEST_ALREADY_IN_FLIGHT','intent':None}
        if self.stopped or self.failed or self.status=='DONE':return {'status':self.status,'intent':None}
        if not self.controller_verified:
            self.status='WAIT_CONTROLLER';self.reason='CONTROLLER_UNVERIFIED'
            return {'status':self.status,'reason':self.reason,'intent':None}
        if not self.station_verified:
            self.status='WAIT_POSITION';self.reason='POSITION_UNVERIFIED'
            return {'status':self.status,'reason':self.reason,'intent':None}
        step=self._steps[self.step_index]
        error=self._precondition(step)
        if error:
            self.status='ERROR';self.reason=error;self.failed=True
            return {'status':self.status,'reason':error,'intent':None}
        now=self.clock_ms() if now_ms is None else now_ms
        error=self._observation_error(observation,context,now)
        if error:
            self.status='WAIT_OBSERVATION';self.reason=error
            return {'status':self.status,'reason':error,'intent':None}
        self.counter+=1
        intent={'session_id':self.session_id,'request_id':self.session_id+':'+str(self.counter),
                'step_index':self.step_index,'step':clone(step),'is_simulated':self.is_simulated,
                'issued_at_ms':now,'observation':clone(observation) if observation else None}
        self.pending=clone(intent)
        self.status='WAIT_RECEIPT';self.reason=None
        if observation is not None:
            self.consumed_frames[str(observation['camera_id'])]=observation['frame_id']
        self._event('INTENT_RESERVED',request_id=intent['request_id'])
        return {'status':self.status,'intent':clone(intent)}

    def _proof_error(self,step,receipt):
        expected_station=(self.start_id if step['action']=='RETURN' else step['dest_station'])
        if step['action']=='PICK':expected_station=step['source_station']
        if receipt.get('action')!=step['action'] or receipt.get('station')!=expected_station:
            return 'COMPLETION_ACTION_OR_STATION_MISMATCH'
        if receipt.get('gripper_empty') is not True:return 'GRIPPER_EMPTY_PROOF_REQUIRED'
        if receipt.get('batch')!=step['batch'] or type(receipt.get('batch')) is not int:return 'COMPLETION_BATCH_MISMATCH'
        expected_count=len(self.carrier)
        if step['action'] in ('PICK','PLACE'):
            for key,field in (('color_id','target_color'),('slot','target_slot'),('layer','layer')):
                if receipt.get(key)!=step[field] or (step[field] is not None and type(receipt.get(key)) is not int):
                    return 'COMPLETION_TARGET_MISMATCH'
            proof='loaded_verified' if step['action']=='PICK' else 'placement_verified'
            if receipt.get(proof) is not True:return 'OBJECT_COMPLETION_PROOF_REQUIRED'
            expected_count+=1 if step['action']=='PICK' else -1
        if type(receipt.get('carrier_count')) is not int or receipt['carrier_count']!=expected_count:
            return 'CARRIER_COUNT_MISMATCH'
        return None

    def _apply_done(self,step,receipt):
        b,c=step['batch'],step['target_color'];key=(b,c)
        if step['action']=='PICK':
            if step['source_station']=='PROCESS':del self.process[step['target_slot']]
            self.objects[key]='CARRIER';self.carrier.append(key);self.picks+=1
        elif step['action']=='PLACE':
            self.carrier.remove(key);self.objects[key]=step['dest_station'];self.placements+=1
            if step['dest_station']=='PROCESS':self.process[step['target_slot']]=key
            else:self.buffer.setdefault(step['target_slot'],[]).append(key)
            self.completed[str(b)+'_'+step['dest_station']]+=1
        else:self.station=self.start_id if step['action']=='RETURN' else step['dest_station']
        self.station_verified=True
        self.step_index+=1
        self.journal.append({'intent':clone(self.pending),'receipt':clone(receipt)})
        self.pending=None
        if self.step_index==len(self._steps):
            complete=(self.picks==12 and self.placements==12 and not self.carrier and not self.process
                and all(v=='BUFFER' for v in self.objects.values()) and all(v==3 for v in self.completed.values())
                and self.station==self.start_id)
            self.status='DONE' if complete else 'ERROR'
            self.failed=not complete
            self.reason=None if complete else 'FINAL_LEDGER_MISMATCH'
        else:self.status='STOPPED' if self.stopped else 'READY'

    def receive(self,receipt,now_ms=None):
        if self.pending is None:return self._reject('NO_PENDING_REQUEST')
        error=self._receipt_source(receipt,self.pending['request_id'])
        if error:return self._reject(error)
        if receipt.get('status') not in ('ACK','BUSY','DONE','ERROR','UNKNOWN'):
            return self._reject('INVALID_RECEIPT_STATUS')
        if self.failed:return self._reject('FAILED_REQUEST_REQUIRES_AUDIT')
        if receipt['status']=='DONE':
            step=self._steps[self.step_index]
            error=self._proof_error(step,receipt)
            if error:
                self.status='UNCERTAIN';self.reason=error
                return self._reject(error)
            self.last_receipt_seq=receipt['seq']
            self._apply_done(step,receipt)
            self._event('STEP_CONFIRMED',request_id=receipt['request_id'])
        else:
            self.last_receipt_seq=receipt['seq']
            if receipt['status']=='ERROR':
                self.status='ERROR';self.failed=True;self.failure_receipt=clone(receipt)
            elif receipt['status']=='UNKNOWN':self.status='UNCERTAIN'
            elif self.stopped:self.status='STOPPED_UNRESOLVED'
            elif self.status!='UNCERTAIN':self.status='WAIT_RECEIPT'
            if receipt['status'] not in ('ACK','BUSY') or self.status!='UNCERTAIN':self.reason=receipt.get('reason')
            self._event('CONTROLLER_STATUS',status=receipt['status'])
        return self.summary()

    def _reject(self,reason):
        self.rejected_receipts+=1
        self._event('RECEIPT_REJECTED',reason=reason)
        result=self.summary();result['receipt_rejection']=reason
        return result

    def poll_timeout(self,now_ms=None):
        now=self.clock_ms() if now_ms is None else now_ms
        if self.pending is not None and not self.failed and self.status!='STOPPED_UNRESOLVED':
            if _elapsed(now,self.pending['issued_at_ms'])>self.timeout_ms:
                self.status='UNCERTAIN';self.reason='CONTROLLER_COMPLETION_TIMEOUT'
        return self.summary()

    def stop(self):
        self.stopped=True
        if self.status not in ('ERROR','DONE'):
            self.status='STOPPED_UNRESOLVED' if self.pending is not None else 'STOPPED'
        return self.summary()

    def placement_evaluations(self,calibrated=False):
        """Evaluate accepted completions without changing the execution ledger.

        The normal summary uses no calibration authority. Explicit calibration
        is an evaluation input, never permission to issue a controller intent.
        """
        if type(calibrated) is not bool:raise ValueError('calibrated must be boolean')
        results=[]
        for entry in self.journal:
            intent,receipt=entry['intent'],entry['receipt'];step=intent['step']
            if step['action']!='PLACE':continue
            if step['layer']==2:
                value={'status':'UNKNOWN','official_points':None,'reason':'STACK_EVALUATION_REQUIRED'}
            else:
                try:value=score_placement_receipt(step['target_slot'],receipt,calibrated=calibrated)
                except (ValueError,TypeError):
                    value={'status':'UNKNOWN','official_points':None,'reason':'INVALID_SCORING_OBSERVATION'}
            value.update(request_id=intent['request_id'],step_index=intent['step_index'],
                         is_simulated=self.is_simulated)
            results.append(value)
        scored=[v['official_points'] for v in results if v['official_points'] is not None]
        return {'scope':'CONFIRMED_PLACEMENT_RECEIPTS_ONLY','is_simulated':self.is_simulated,
            'denominator':len(results),'scored':len(scored),'unknown':len(results)-len(scored),
            'rule_points':sum(scored) if results and len(scored)==len(results) else None,
            'evaluations':results}

    def summary(self):
        scoring=self.placement_evaluations()
        scoring.pop('evaluations')
        return {'session_id':self.session_id,'mode':self.mode,'is_simulated':self.is_simulated,
                'status':self.status,'reason':self.reason,'task':self.task.payload,
                'start_id':self.start_id,'station':self.station,'station_verified':self.station_verified,
                'step_index':self.step_index,'planned_steps':len(self._steps),
                'confirmed_steps':self.step_index,'counters':{'picks':self.picks,'placements':self.placements},
                'completed':dict(self.completed),'carrier':[list(k) for k in self.carrier],
                'pending':clone(self.pending),'controller_verified':self.controller_verified,
                'failed':self.failed,
                'rejected_receipts':self.rejected_receipts,'placement_scoring':scoring}

    def snapshot(self):
        return {'schema_version':1,'session_id':self.session_id,'mode':self.mode,'task':self.task.payload,
                'capacity':self.capacity,'start_id':self.start_id,'timeout_ms':self.timeout_ms,
                'site_receipt':clone(self.site_receipt),'journal':clone(self.journal),
                'pending':clone(self.pending),'last_receipt_seq':self.last_receipt_seq,
                'consumed_frames':dict(self.consumed_frames),'counter':self.counter,
                'stopped':self.stopped,'status':self.status,'reason':self.reason,'failed':self.failed,
                'failure_receipt':clone(self.failure_receipt)}

    @classmethod
    def restore(cls,snapshot,controller_verified=False,clock_ms=None):
        if snapshot.get('schema_version')!=1:raise ValueError('Unknown workflow snapshot')
        session=cls(snapshot['task'],snapshot['session_id'],snapshot['mode'],snapshot['capacity'],
                    snapshot['start_id'],controller_verified,snapshot['timeout_ms'],clock_ms)
        # Replay accepted completion data only; never publish or execute an intent.
        session.site_receipt=clone(snapshot['site_receipt'])
        if session.site_receipt is not None:
            r=session.site_receipt
            if (r.get('session_id')!=session.session_id or r.get('request_id')!=session.session_id+':POSITION'
                    or r.get('is_simulated') is not session.is_simulated or r.get('status')!='DONE'
                    or r.get('action')!='POSITION' or r.get('station')!='QR' or r.get('gripper_empty') is not True
                    or type(r.get('seq')) is not int or not 0<=r['seq']<=4294967295
                    or r.get('start_id')!=session.start_id):raise ValueError('Invalid position history')
            session.station='QR';session.station_verified=True;session.last_receipt_seq=r['seq']
        elif not session.is_simulated and snapshot['journal']:
            raise ValueError('Real position history missing')
        for entry in snapshot['journal']:
            receipt=entry['receipt'];intent=entry['intent']
            if session.step_index>=len(session._steps):raise ValueError('Excess completions')
            index=session.step_index
            if (receipt.get('session_id')!=session.session_id or receipt.get('request_id')!=session.session_id+':'+str(index+1)
                    or receipt.get('is_simulated') is not session.is_simulated or receipt.get('status')!='DONE'
                    or type(receipt.get('seq')) is not int or receipt['seq']<=session.last_receipt_seq):
                raise ValueError('Invalid completion history')
            step=session._steps[index]
            if (intent.get('step')!=clone(step) or intent.get('step_index')!=index
                    or intent.get('request_id')!=receipt['request_id'] or intent.get('session_id')!=session.session_id
                    or intent.get('is_simulated') is not session.is_simulated
                    or type(intent.get('issued_at_ms')) is not int):raise ValueError('Invalid intent history')
            observation=intent.get('observation')
            if session._observation_error(observation,None,intent['issued_at_ms'],historical=True):
                raise ValueError('Invalid historical observation')
            if observation is not None:
                session.consumed_frames[str(observation['camera_id'])]=observation['frame_id']
            if session._precondition(step) or session._proof_error(step,receipt):raise ValueError('Invalid completion ledger')
            session.pending=clone(intent)
            session.last_receipt_seq=receipt['seq'];session._apply_done(step,receipt)
        if (type(snapshot['counter']) is not int or snapshot['counter']!=session.step_index+(snapshot['pending'] is not None)
                or type(snapshot['last_receipt_seq']) is not int or not -1<=snapshot['last_receipt_seq']<=4294967295
                or snapshot['last_receipt_seq']<session.last_receipt_seq):
            raise ValueError('Inconsistent workflow sequence')
        session.counter=snapshot['counter'];session.last_receipt_seq=snapshot['last_receipt_seq']
        session.pending=clone(snapshot['pending'])
        if session.pending is not None:
            p=session.pending
            if (session.step_index>=len(session._steps) or p.get('session_id')!=session.session_id
                    or p.get('request_id')!=session.session_id+':'+str(session.counter)
                    or p.get('step_index')!=session.step_index or p.get('step')!=clone(session._steps[session.step_index])
                    or p.get('is_simulated') is not session.is_simulated
                    or type(p.get('issued_at_ms')) is not int):raise ValueError('Invalid pending intent')
            observation=p.get('observation')
            if session._observation_error(observation,None,p['issued_at_ms'],historical=True):raise ValueError('Invalid pending observation')
            if observation is not None:session.consumed_frames[str(observation['camera_id'])]=observation['frame_id']
        if session.consumed_frames!=snapshot['consumed_frames']:raise ValueError('Invalid consumed-frame history')
        session.failed=snapshot['failed'] is True
        session.failure_receipt=clone(snapshot['failure_receipt'])
        if not session.failed and (snapshot['status']=='ERROR' or session.failure_receipt is not None):
            raise ValueError('Failure state cannot be cleared by restore')
        if session.failure_receipt is not None:
            failure=session.failure_receipt
            if (session.pending is None or failure.get('session_id')!=session.session_id
                    or failure.get('request_id')!=session.pending['request_id'] or failure.get('status')!='ERROR'
                    or failure.get('is_simulated') is not session.is_simulated):raise ValueError('Invalid failure history')
        session.stopped=True
        session.station_verified=False
        if session.failed:session.status='ERROR'
        elif session.step_index==len(session._steps):session.status='DONE'
        elif session.pending is not None:session.status='STOPPED_UNRESOLVED'
        else:session.status='STOPPED'
        session.reason='RESTORED_REQUIRES_RECONCILIATION' if session.pending else 'RESTORED_REQUIRES_POSITION_AND_RESUME'
        return session

    def resume(self):
        if self.failed:raise ValueError('Failed workflow requires audit, not replay')
        if self.pending is not None:raise ValueError('Resolve pending intent before resume')
        if self.status=='DONE':return self.summary()
        if not self.controller_verified:raise ValueError('Controller must be verified again')
        if not self.station_verified:raise ValueError('Position must be verified again')
        self.stopped=False;self.status='READY';self.reason=None
        return self.summary()
