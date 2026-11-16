import os,time,gc,json,sys
import image
from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager
from nonai2d import CSC
s=None;c=None;r={}
try:
 Display.init(Display.ST7701,width=800,height=480,to_ide=True)
 s=Sensor(id=2,width=1920,height=1080,fps=30);s.reset()
 s.set_framesize(width=640,height=360,crop=False)
 s.set_pixformat(Sensor.RGBP888)
 MediaManager.init();s.run()
 c=CSC(CSC.PIXEL_FORMAT_RGB_565,max_width=640,max_height=360)
 start=time.ticks_ms();count=0
 while time.ticks_diff(time.ticks_ms(),start)<20000:
  os.exitpoint()
  f=s.snapshot(dump_frame=True,auto_reboot=False)
  if count==0:
   print('VIDEO_INFO',type(f),dir(f))
   if hasattr(f,'v_frame'):print('VIDEO_FRAME',type(f.v_frame),dir(f.v_frame))
  converted=c.convert(f)
  if count==60:converted.save('/sdcard/cdiag1116/csc-dump.bmp')
  Display.show_image(converted,x=80,y=60)
  f=converted=None;gc.collect()
  count+=1
 r.update(status='COMPLETE',frames=count)
except BaseException as e:
 r.update(status='FAILED',error=str(e));sys.print_exception(e)
finally:
 if s:s.stop()
 Display.deinit()
 if c:c.destroy()
 MediaManager.deinit()
 with open('/sdcard/cdiag1116/csc-dump.json','w') as out:json.dump(r,out)
 print('CSC_DUMP_DONE '+json.dumps(r))

