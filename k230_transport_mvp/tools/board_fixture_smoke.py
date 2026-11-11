"""CanMV file-fixture tests, deliberately separated from live camera evidence."""
import sys
if sys.implementation.name!='micropython':
    raise SystemExit('Run explicitly using CanMV on the K230.')
import image
import json
import time
sys.path.insert(0,'/sdcard/transport_mvp')
from k230.vision_adapter import VisionAdapter
from ports.protocol import encode_observation,StreamParser

ROOT='/sdcard/transport_mvp'
profile={'width':640,'height':320,'crop':False,'hmirror':False,'vflip':False,
         'rotation':0,'pose_id':'FILE_FIXTURE_ONLY'}
thresholds={int(k):v for k,v in json.load(open(ROOT+'/fixtures/thresholds.json')).items()}
layout=json.load(open(ROOT+'/fixtures/synthetic-ring-layout.json'))
if layout.get('is_simulated') is not True:
    raise ValueError('Fixture geometry must be explicitly simulated')
center_gates={int(k):v for k,v in layout['center_gates'].items()}
cfg={'CAMERA_IDS':(2,),'DESKTOP_SINGLE_CAMERA':True,
     'MODE_CAMERA':{'QR':2,'COLOR':2,'RING':2,'LANE':2,'TOP':2},
     'COLOR_THRESHOLDS':thresholds,'COLOR_ROI':(1,1,638,318),'COLOR_REFERENCE':(320,160),
     'COLOR_MIN_AREA':40,'COLOR_MAX_AREA':60000,
     'PLANE_BY_MODE_STATION':{('COLOR','RAW'):'RAW_TOP',('TOP','BUFFER'):'BUFFER_TOP',('RING','PROCESS'):'PROCESS_PLANE'},
     'TOP_CONFIG':{'BUFFER':{'thresholds':thresholds,'roi':(1,1,638,318),'reference':(320,160),'min_area':40,'max_area':60000}},
     'RING_CONFIG':{'PROCESS':{'roi':(0,0,640,320),'camera_id':2,'image_profile':profile,
                              'pose_id':'FILE_FIXTURE_ONLY','ring_id_order':(1,2,3),
                              'verified':True,'min_spacing_px':100,
                              'center_gates':center_gates,
                              'parameters':{'threshold':2000,'r_min':18,'r_max':42,'r_step':2}}},
     'LANE_ROI':(0,80,640,220),'LANE_THRESHOLD':(100,160),'LANE_MIN_WIDTH_PX':200,
     'LANE_MAX_WIDTH_PX':280,'CALIBRATIONS':{}}
vision=VisionAdapter(cfg)
results=[]
frame_id=0
for file,mode,targets,station in [('phone-padded.jpg','COLOR',[1,2,3,4,6],'RAW'),
                                 ('phone-padded.jpg','TOP',[6],'BUFFER'),
                                 ('synthetic-rings.jpg','RING',[1,2,3],'PROCESS'),
                                 ('synthetic-lane.jpg','LANE',[None],None)]:
    for target in targets:
        # Reload: conversion to grayscale can mutate the image in this firmware.
        img=image.Image(ROOT+'/fixtures/'+file).to_rgb565()
        frame_id+=1
        frame={'image':img,'camera_id':2,'frame_id':frame_id,'timestamp_ms':frame_id*40,
               'is_simulated':True,'image_profile':dict(profile)}
        obs=vision.detect(mode,frame,target_id=target,station=station)
        wire=encode_observation(frame_id,obs)
        decoded=StreamParser().feed(wire,frame_id*40)
        results.append({'file':file,'observation':obs,'wire_decoded':len(decoded)==1})
report={'test':'NATIVE_CANMV_FILE_VISION','source':'FILE_FIXTURE_NOT_CAMERA',
        'is_simulated':True,'motors_enabled':False,'results':results,
        'passed_pixels':sum(r['observation']['u_px'] is not None for r in results),
        'expected_pixels':len(results),
        'all_mm_denied':all(r['observation']['x_mm'] is None and r['observation']['target_pose_valid'] is False for r in results)}
with open(ROOT+'/native_fixture_report.json','w') as output:
    output.write(json.dumps(report))
print('KMVP_NATIVE_FIXTURES:'+json.dumps(report))
