"""Explicit bounded CSI2/CSI1 probe after operator-confirmed power-cycle; skips CSI0."""
import sys,os,time,json
if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230.')
from media.sensor import Sensor
from media.media import MediaManager
ROOT='/sdcard/transport_mvp/'
REPORT='role_swap_after_power_cycle_probe.json'
sensors={};rows=[];ready=False
started=time.ticks_ms()
try:
    for cid in (2,1):
        try:
            sensor=Sensor(id=cid,width=1920,height=1080,fps=30)
            sensors[cid]=sensor
            sensor.reset();sensor.set_framesize(width=320,height=240);sensor.set_pixformat(Sensor.RGB565)
            rows.append({'camera_id':cid,'init':'PASS','attempts':0,'frames':0,'errors':[]})
        except Exception as error:
            rows.append({'camera_id':cid,'init':'FAIL','attempts':0,'frames':0,'errors':[str(error)]})
            if cid in sensors:
                try:sensors[cid].stop(is_del=True)
                except Exception:pass
                del sensors[cid]
    if sensors:
        MediaManager.init();ready=True
        sensors[next(iter(sensors))].run()
        for row in rows:
            cid=row['camera_id']
            if cid not in sensors:continue
            for index in range(30):
                os.exitpoint();row['attempts']+=1
                try:
                    frame=sensors[cid].snapshot(timeout=2000 if index==0 else 150,auto_reboot=False)
                    if frame is None:raise RuntimeError('Empty camera frame')
                    row['frames']+=1
                    if index==29:
                        name='role-swap-after-power-csi%d-live.jpg' % cid
                        frame.save(ROOT+name,quality=95);row['image']=name
                except Exception as error:row['errors'].append(str(error))
finally:
    for sensor in sensors.values():
        try:sensor.stop()
        except Exception:pass
    if ready:MediaManager.deinit()
    report={'test':'CAMERA_ROLE_PROBE_AFTER_USER_POWER_CYCLE','source':'REAL_CAMERA',
        'operator_reported_power_cycle':True,'results':rows,'initialization_order':[2,1],
        'elapsed_ms':time.ticks_diff(time.ticks_ms(),started),'quarantined':[0],
        'object_camera_id':1,'front_camera_id':2,'motors_enabled':False}
    with open(ROOT+REPORT,'w') as stream:stream.write(json.dumps(report))
    print('ROLE_CAMERA_POWER_PROBE:'+json.dumps(report))
