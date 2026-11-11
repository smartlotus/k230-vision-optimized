"""Freeze one real CSI2 frame and its native LAB grid for offline ROI labeling."""
import sys,os,time,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
from media.sensor import Sensor
from media.media import MediaManager
sensor=None
ready=False
try:
    sensor=Sensor(id=2,width=1920,height=1080,fps=30)
    sensor.reset();sensor.set_framesize(width=320,height=240);sensor.set_pixformat(Sensor.RGB565)
    MediaManager.init();ready=True;sensor.run()
    for i in range(45):
        os.exitpoint();sensor.snapshot(timeout=2000 if i==0 else 150,auto_reboot=False)
    frame=sensor.snapshot(timeout=150,auto_reboot=False)
    frame.save('/sdcard/transport_mvp/block-frozen-live.jpg',quality=95)
    import image
    grid=[]
    for y in range(2,240,4):
        os.exitpoint()
        grid.append([image.rgb_to_lab(*frame.get_pixel(x,y)) for x in range(2,320,4)])
    report={'source':'FROZEN_REAL_CSI2_RGB565_NATIVE_LAB','camera_id':2,'width':320,'height':240,
        'x0':2,'y0':2,'stride':4,'lab_grid':grid,'distance_cm':None,
        'camera_mount_fixed':False,'jpeg':'block-frozen-live.jpg','motors_enabled':False}
    with open('/sdcard/transport_mvp/block_frozen_lab.json','w') as output:output.write(json.dumps(report))
    print('BLOCK_FROZEN_SAVED')
finally:
    if sensor:sensor.stop()
    if ready:MediaManager.deinit()
