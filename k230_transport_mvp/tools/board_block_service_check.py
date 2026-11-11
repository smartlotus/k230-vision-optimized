"""Real dual-camera -> task-mode observation -> protocol, no UART pins or motion."""
import sys,os,time,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
sys.path.insert(0,'/sdcard/transport_mvp')
from k230.block_profile import make_config
from k230.camera_manager import CameraManager
from k230.vision_adapter import VisionAdapter
from ports.protocol import encode_observation,StreamParser
config=make_config()
camera=CameraManager(config)
vision=VisionAdapter(config)
report={'test':'REAL_DUAL_BLOCK_OBSERVATION_PROTOCOL','observations':[], 'errors':[],
        'control_transport':'IN_PROCESS_CODEC_NOT_ELECTRICAL_UART','motors_enabled':False}
try:
    camera.start()
    for i in range(45):
        os.exitpoint();camera.get_frame(2)
    for color in (1,2,3,4,5,6):
        for i in range(3):
            os.exitpoint()
            obs=vision.detect('COLOR',camera.get_frame(2),target_id=color,station='RAW')
        wire=encode_observation(color,obs)
        decoded=StreamParser().feed(wire,time.ticks_ms())
        assert len(decoded)==1 and decoded[0]['camera_id']==2
        assert obs['is_simulated'] is False and obs['x_mm'] is None and obs['y_mm'] is None
        assert obs['target_pose_valid'] is False
        report['observations'].append(obs)
    for mode,target,station in (('QR',None,'QR'),('LANE',None,None),('TOP',1,'BUFFER'),('RING',1,'PROCESS')):
        cid=camera.camera_for_mode(mode)
        obs=vision.detect(mode,camera.get_frame(cid),target_id=target,station=station)
        assert obs['camera_id']==(1 if mode in ('QR','LANE') else 2)
        assert obs['target_pose_valid'] is False
        report['observations'].append(obs)
    report['pipeline_pass']=True
except Exception as error:
    report['pipeline_pass']=False
    report['errors'].append(str(error));sys.print_exception(error)
finally:
    camera.close();report['cleanup_errors']=camera.close_errors
    with open('/sdcard/transport_mvp/block_service_check.json','w') as output:output.write(json.dumps(report))
    print('BLOCK_SERVICE_RESULT:'+json.dumps(report))
