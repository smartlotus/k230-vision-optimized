"""Freeze the active object-camera scene and native LAB; no inferred calibration."""
import sys,os,time,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
from media.sensor import Sensor
from media.media import MediaManager
import image
sys.path.insert(0,'/sdcard/transport_mvp')
from k230.camera_roles import OBJECT_CAMERA_ID
ROOT='/sdcard/transport_mvp/'
PREFIX='csi%d-top-frozen01' % OBJECT_CAMERA_ID
sensor=None;ready=False
try:
    sensor=Sensor(id=OBJECT_CAMERA_ID,width=1920,height=1080,fps=30)
    sensor.reset();sensor.set_framesize(width=320,height=240);sensor.set_pixformat(Sensor.RGB565)
    MediaManager.init();ready=True;sensor.run()
    for index in range(45):
        os.exitpoint();sensor.snapshot(timeout=2000 if index==0 else 150,auto_reboot=False)
    frame=sensor.snapshot(timeout=150,auto_reboot=False)
    captured_ms=time.ticks_ms()
    frame.save(ROOT+PREFIX+'-live.jpg',quality=95)
    lossless=None;save_error=None
    try:
        frame.save(ROOT+PREFIX+'-live.bmp');lossless=PREFIX+'-live.bmp'
    except Exception as error:save_error=str(error)
    grid=[]
    for y in range(1,240,2):
        os.exitpoint()
        grid.append([image.rgb_to_lab(*frame.get_pixel(x,y)) for x in range(1,320,2)])
    report={'source':'FROZEN_REAL_CSI%d_RGB565_NATIVE_LAB' % OBJECT_CAMERA_ID,'camera_id':OBJECT_CAMERA_ID,'width':320,'height':240,
        'x0':1,'y0':1,'stride':2,'lab_grid':grid,'captured_ms':captured_ms,'sample_index':46,
        'image_profile':{'width':320,'height':240,'crop':False,'hmirror':False,'vflip':False,
                         'rotation':0,'pose_id':None},
        'distance_cm':None,'camera_mount_fixed':False,'setup_state':'USER_REPORTS_TOP_FACES_POSITIONED',
        'jpeg':PREFIX+'-live.jpg','lossless':lossless,'lossless_save_error':save_error,'motors_enabled':False}
    with open(ROOT+PREFIX+'-lab-grid.json','w') as stream:stream.write(json.dumps(report))
    print('TOP_FROZEN_SAVED:'+json.dumps({'jpeg':report['jpeg'],'lossless':lossless,
        'grid':PREFIX+'-lab-grid.json','captured_ms':captured_ms,'motors_enabled':False}))
finally:
    if sensor:sensor.stop()
    if ready:MediaManager.deinit()
