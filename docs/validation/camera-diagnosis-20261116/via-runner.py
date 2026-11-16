import os,time,gc,json,sys
import image
from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager
s=None
r={'kind':'rgbp_via_rgb888'}
try:
 Display.init(Display.ST7701,width=800,height=480,to_ide=True)
 s=Sensor(id=2,width=1920,height=1080,fps=30);s.reset()
 s.set_framesize(width=640,height=360,crop=False)
 s.set_pixformat(Sensor.RGBP888)
 MediaManager.init();s.run()
 start=time.ticks_ms();count=0
 while time.ticks_diff(time.ticks_ms(),start)<25000:
  os.exitpoint()
  f=s.snapshot(auto_reboot=False)
  packed=f.to_rgb888()
  converted=packed.to_rgb565()
  if count==30:
   with open('/sdcard/cdiag1116/via-source.raw','wb') as out:out.write(f.bytearray())
   with open('/sdcard/cdiag1116/via-packed.raw','wb') as out:out.write(packed.bytearray())
   converted.save('/sdcard/cdiag1116/via-converted.bmp')
  Display.show_image(converted,x=80,y=60)
  f=packed=converted=None;gc.collect()
  count+=1
 r.update(status='COMPLETE',frames=count)
except BaseException as e:
 r.update(status='FAILED',error=str(e));sys.print_exception(e)
finally:
 if s:s.stop()
 Display.deinit()
 MediaManager.deinit()
 with open('/sdcard/cdiag1116/via.json','w') as out:json.dump(r,out)
 print('VIA_DONE '+json.dumps(r))

