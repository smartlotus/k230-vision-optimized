"""Controlled software faults with real camera frames, then continuous preview."""
import sys,json,time
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
ROOT='/sdcard/transport_mvp/'
sys.path.insert(0,ROOT)
import media.sensor as sensor_module
import k230.visual_inspector as inspector_module
from k230.capture_io import CaptureIO
from run_block_preview import main
NativeSensor=sensor_module.Sensor
NativeInspector=inspector_module.VisualInspector
profile=json.load(open(ROOT+'vision_pixel_profile.json'))
storage=CaptureIO();reports=[];control={}


class FaultSensor(NativeSensor):
    def __init__(self,*args,**kwargs):
        self.test_camera_id=kwargs.get('id')
        super().__init__(*args,**kwargs)

    def snapshot(self,*args,**kwargs):
        if self.test_camera_id==1 and control.get('armed'):
            if control['permanent'] or control['injected']<25:
                time.sleep_ms(55)
                control['injected']+=1
                raise RuntimeError('INJECTED_CSI1_TIMEOUT')
        return super().snapshot(*args,**kwargs)


class InspectorSpy(NativeInspector):
    def process(self,frame):
        result=super().process(frame)
        control['processed']+=1
        if control['processed']==2:control['armed']=True
        return result


sensor_module.Sensor=FaultSensor
inspector_module.VisualInspector=InspectorSpy
try:
    for permanent in (False,True):
        control={'armed':False,'injected':0,'processed':0,'permanent':permanent}
        report=main(max_frames=12,max_elapsed_ms=2000 if permanent else 10000,
            inspection_mode='TOP',inspection_target=5,inspection_station='BUFFER',pixel_profile=profile,
            control_path=None,capture_path=None,qr_enabled=False,
            report_path=ROOT+('fault-permanent-20261102.json' if permanent else 'fault-recovery-20261102.json'))
        checks={'auxiliary_continued':report['counts']['2']>=3,
            'mode_preserved':report['inspection']['mode']=='TOP',
            'original_error_preserved':report['camera_health']['1']['first_error']=='INJECTED_CSI1_TIMEOUT',
            'fault_count_exact':report['errors']['1']==control['injected']}
        if permanent:
            checks.update(deadline_enforced=report['stop_reason']=='TIME_LIMIT' and report['frames']==2,
                stale_observation_cleared=report['observation'] is None,
                incomplete_reported=report['completed_requested_frames'] is False)
        else:
            checks.update(recovered=report['frames']==12 and report['errors']['1']==25,
                consecutive_cleared=report['camera_health']['1']['consecutive_errors']==0)
        reports.append({'permanent':permanent,'injected_failures':control['injected'],'checks':checks,'report':report})
        print('CAPTURE_FAULT_TEST:'+json.dumps({'permanent':permanent,'checks':checks}))
finally:
    sensor_module.Sensor=NativeSensor
    inspector_module.VisualInspector=NativeInspector
    result={'test':'CAPTURE_FAULT_PATH_INJECTION','source':'INJECTED_FAILURE_WITH_REAL_CAMERA_FRAMES',
        'native_camera_fault_claim':False,'cases':reports,'motors_enabled':False,
        'passed':len(reports)==2 and all(all(c['checks'].values()) for c in reports)}
    storage.save_json(result,ROOT+'capture_fault_check_20261102.json')
    print('CAPTURE_FAULT_TEST_RESULT:'+json.dumps({'passed':result['passed'],'cases':len(reports)}))

if len(reports)==2:
    print('CAPTURE_FAULT_RESTORE_CONTINUOUS')
    main(inspection_mode='TOP',inspection_target=5,inspection_station='BUFFER',pixel_profile=profile,
        capture_path=None,control_path=ROOT+'vision_control.json',qr_enabled=False,
        report_path=ROOT+'top_post_fault_check_20261102.json')
