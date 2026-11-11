"""Bounded CanMV camera/adapter smoke; run explicitly, never as boot.py."""
import sys
if sys.implementation.name!='micropython':
    raise SystemExit('Run this bounded smoke only on the K230 using CanMV.')
import os
import time
import json
import gc

ROOT='/sdcard/kmvp_20261031_v1'
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
from k230.camera_manager import CameraManager
from k230.vision_adapter import VisionAdapter
from k230 import config

config.PREVIEW=True
camera=CameraManager(config)
vision=VisionAdapter(config)
report={'is_simulated':False,'motors_enabled':False,'uart_enabled':False,
        'test':'CSI2_CAPTURE_AND_UNCALIBRATED_ADAPTER_SMOKE','finished':False,
        'observations':[],'capture_ms':[],'processing_ms':[],'errors':[]}
try:
    print('KMVP_PHASE:camera_start')
    camera.start()
    print('KMVP_PHASE:capture_begin')
    for mode,target,station in [('QR',None,'QR'),('COLOR',1,'RAW'),('RING',1,'PROCESS'),
                                ('LANE',None,None),('TOP',1,'BUFFER')]:
        for index in range(4):
            os.exitpoint()
            before=time.ticks_ms()
            frame=camera.get_frame(camera.camera_for_mode(mode))
            report['capture_ms'].append(time.ticks_diff(time.ticks_ms(),before))
            before=time.ticks_ms()
            result=vision.detect(mode,frame,target_id=target,station=station)
            report['processing_ms'].append(time.ticks_diff(time.ticks_ms(),before))
            report['observations'].append(result)
            print('KMVP_OBS:'+json.dumps(result))
            if frame is not None:
                frame['image'].draw_string_advanced(5,5,20,mode+' '+result['status'],color=(255,255,255))
                camera.show_frame(frame)
            time.sleep_ms(30)
    report['finished']=True
except Exception as error:
    report['errors'].append(str(error))
    sys.print_exception(error)
finally:
    print('KMVP_PHASE:camera_close')
    camera.close()
    report['cleanup_errors']=camera.close_errors
    gc.collect()
    report['heap_free_bytes']=gc.mem_free()
    report['mean_capture_ms']=sum(report['capture_ms'])/len(report['capture_ms']) if report['capture_ms'] else None
    report['mean_processing_ms']=sum(report['processing_ms'])/len(report['processing_ms']) if report['processing_ms'] else None
    report['capture_status']='PASS' if report['finished'] and not report['errors'] and not report['cleanup_errors'] else 'FAIL'
    with open(ROOT+'/camera_smoke.json','w') as output:
        output.write(json.dumps(report))
    print('KMVP_RESULT:'+json.dumps({k:report[k] for k in ['finished','capture_status','mean_capture_ms','mean_processing_ms','heap_free_bytes','errors','cleanup_errors']}))
