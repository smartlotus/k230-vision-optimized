import copy
import unittest
from core.workflow_session import WorkflowSession
from sim.run_workflow import run_workflow,AsyncSimController
from sim.field_sim import FieldSim
from sim.motion_sim import MotionSim
from core.task_parser import parse_task
from test_mission_sim import legal_payloads


PAYLOAD='156+123+516+231'


def done(session,intent,seq=2):
    s=intent['step']
    return {'session_id':session.session_id,'request_id':intent['request_id'],'seq':seq,
            'status':'DONE','is_simulated':session.is_simulated,'action':s['action'],
            'station':session.start_id if s['action']=='RETURN' else s['source_station'] if s['action']=='PICK' else s['dest_station'],
            'batch':s['batch'],'color_id':s['target_color'],'slot':s['target_slot'],'layer':s['layer'],
            'gripper_empty':True,'carrier_count':len(session.carrier)+(1 if s['action']=='PICK' else -1 if s['action']=='PLACE' else 0),
            'loaded_verified':True,'placement_verified':True}


class WorkflowTests(unittest.TestCase):
    def test_whole_two_batch_ack_then_done_matches_independent_field(self):
        r=run_workflow()
        self.assertEqual(r['summary']['status'],'DONE')
        self.assertTrue(r['field_complete'])
        self.assertEqual(r['summary']['counters'],{'picks':12,'placements':12})
        self.assertEqual(r['summary']['confirmed_steps'],31)
        self.assertEqual(r['summary']['station'],'START_1')
        self.assertEqual(r['controller_submissions'],31)
        self.assertEqual(len(r['controller_receipts']),62)
        self.assertEqual([t['slot'] for t in r['transitions'] if t['batch']==2 and t['dest']=='BUFFER'],[2,1,3])
        scoring=r['summary']['placement_scoring']
        self.assertEqual((scoring['denominator'],scoring['scored'],scoring['unknown']),(12,0,12))
        self.assertIsNone(scoring['rule_points'])

    def test_scoring_consumes_postplacement_zone_without_changing_workflow_ledger(self):
        session=WorkflowSession.restore(run_workflow()['workflow_snapshot'])
        entry=next(e for e in session.journal if e['intent']['step']['action']=='PLACE')
        receipt=entry['receipt'];slot=entry['intent']['step']['target_slot']
        receipt.update(observed_slot=slot,scoring_zone=1,placement_state='IN_RING',
                       stable=True,tilted=False,score_observation_verified=True)
        before=session.snapshot()
        scoring=session.placement_evaluations(calibrated=True)
        self.assertEqual(scoring['scored'],1)
        self.assertEqual(next(e for e in scoring['evaluations'] if e['official_points'] is not None)['official_points'],15)
        self.assertEqual(session.snapshot(),before)
        self.assertIsNone(scoring['rule_points'])
        receipt['scoring_zone']=99
        scoring=session.placement_evaluations(calibrated=True)
        self.assertEqual(scoring['scored'],0)
        self.assertEqual(session.placements,12)

    def test_twenty_tasks_three_capacities_two_starts(self):
        for payload in legal_payloads():
            for capacity in (1,2,3):
                for start in ('START_1','START_2'):
                    with self.subTest(payload=payload,capacity=capacity,start=start):
                        r=run_workflow(payload,capacity,start)
                        self.assertEqual(r['summary']['status'],'DONE')
                        self.assertTrue(r['field_complete'])
                        self.assertLessEqual(r['max_carrier_occupancy'],capacity)

    def test_all_existing_faults_cannot_finish_or_replay(self):
        from sim.run_mission import FAULTS
        for fault in FAULTS:
            with self.subTest(fault=fault):
                r=run_workflow(inject=fault)
                self.assertNotEqual(r['summary']['status'],'DONE')
                self.assertFalse(r['field_complete'])
                if fault in ('ack-loss','timeout'):
                    self.assertIsNotNone(r['summary']['pending'])
                    self.assertEqual(r['summary']['counters']['picks'],0)

    def test_ack_timeout_late_ack_and_done_never_submit_twice(self):
        session=WorkflowSession(PAYLOAD,'x',mode='SIM',receipt_timeout_ms=50)
        intent=session.issue(now_ms=0)['intent']
        ack={'session_id':'x','request_id':intent['request_id'],'seq':1,'status':'ACK','is_simulated':True}
        session.receive(ack)
        self.assertEqual(session.step_index,0)
        session.poll_timeout(100)
        self.assertEqual(session.status,'UNCERTAIN')
        self.assertIsNone(session.issue(now_ms=101)['intent'])
        session.receive(dict(ack,seq=2))
        self.assertEqual(session.status,'UNCERTAIN')
        session.receive(done(session,intent,3))
        self.assertEqual(session.step_index,1)
        session.receive(done(session,intent,4))
        self.assertEqual(session.step_index,1)

    def test_bad_source_session_request_proof_and_replay_do_not_advance(self):
        session=WorkflowSession(PAYLOAD,'x',mode='SIM')
        intent=session.issue(now_ms=0)['intent'];receipt=done(session,intent)
        for change in ({'is_simulated':False},{'session_id':'other'},{'request_id':'x:99'},
                       {'carrier_count':3},{'gripper_empty':False},{'batch':True},{'station':'BUFFER'}):
            session.receive(dict(receipt,**change))
            self.assertEqual(session.step_index,0)
        session.receive(receipt)
        self.assertEqual(session.step_index,1)

    def test_stop_preserves_pending_and_late_completion_does_not_resume(self):
        session=WorkflowSession(PAYLOAD,'x',mode='SIM')
        intent=session.issue(now_ms=0)['intent']
        session.stop()
        self.assertIsNone(session.issue(now_ms=100)['intent'])
        session.receive(done(session,intent))
        self.assertEqual(session.status,'STOPPED')
        self.assertEqual(session.step_index,1)
        self.assertIsNone(session.issue(now_ms=101)['intent'])

    def test_controller_error_remains_terminal_after_stop_and_restore(self):
        session=WorkflowSession(PAYLOAD,'x',mode='SIM')
        intent=session.issue(now_ms=0)['intent']
        session.receive({'session_id':'x','request_id':intent['request_id'],'seq':1,'status':'ERROR','is_simulated':True})
        session.stop()
        restored=WorkflowSession.restore(session.snapshot(),controller_verified=True)
        restored.receive(done(restored,intent,2))
        self.assertEqual(restored.step_index,0)
        self.assertTrue(restored.failed)
        with self.assertRaises(ValueError):restored.resume()

    def test_restore_rebuilds_ledger_and_keeps_uncertain_request(self):
        r=run_workflow(inject='ack-loss')
        restored=WorkflowSession.restore(r['workflow_snapshot'],controller_verified=True)
        self.assertEqual(restored.step_index,r['summary']['step_index'])
        self.assertEqual(restored.status,'STOPPED_UNRESOLVED')
        self.assertEqual(restored.pending,r['summary']['pending'])
        self.assertIsNone(restored.issue(now_ms=9999)['intent'])
        self.assertEqual(restored.picks,0)

    def test_completed_restore_and_corrupted_history(self):
        snapshot=run_workflow()['workflow_snapshot']
        restored=WorkflowSession.restore(snapshot)
        self.assertEqual(restored.status,'DONE')
        self.assertEqual(restored.picks,12)
        for mutate in ('slot','frame','counter'):
            bad=copy.deepcopy(snapshot)
            if mutate=='slot':bad['journal'][1]['receipt']['color_id']=5
            elif mutate=='frame':bad['consumed_frames']={}
            else:bad['counter']=0
            with self.assertRaises(ValueError):WorkflowSession.restore(bad)

    def test_live_default_and_simulated_receipts_cannot_report_real_progress(self):
        session=WorkflowSession(PAYLOAD,'live')
        self.assertIsNone(session.issue(now_ms=0)['intent'])
        fake={'session_id':'live','request_id':'live:POSITION','seq':1,'status':'DONE',
              'is_simulated':True,'action':'POSITION','station':'QR','start_id':'START_1','gripper_empty':True}
        session.verify_position(fake)
        self.assertEqual(session.step_index,0)
        self.assertFalse(session.station_verified)
        self.assertEqual(session.rejected_receipts,1)

    def test_live_unmeasured_context_never_reserves_a_controller_intent(self):
        session=WorkflowSession(PAYLOAD,'live',controller_verified=True)
        position={'session_id':'live','request_id':'live:POSITION','seq':1,'status':'DONE',
                  'is_simulated':False,'action':'POSITION','station':'QR','start_id':'START_1','gripper_empty':True}
        session.verify_position(position)
        obs={'kind':'LANE','target_id':None,'camera_id':1,'station':'QR','status':'OK','stable':True,
             'is_simulated':False,'frame_id':1,'timestamp_ms':100}
        result=session.issue(obs,now_ms=100)
        self.assertIsNone(result['intent'])
        self.assertEqual(result['reason'],'MOTION_NOT_AUTHORIZED')
        self.assertEqual(session.counter,0)


if __name__=='__main__':unittest.main()
