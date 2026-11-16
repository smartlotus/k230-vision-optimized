import os,time,json,sys,gc
import image
from media.sensor import Sensor,CAM_CHN_ID_0,CAM_CHN_ID_1,CAM_CHN_ID_2
from media.display import Display
from media.media import MediaManager
root='/sdcard/cdiag1116'
s=None
r={'kind':'three_channel_formats','frames':0}
try:
 Display.init(Display.ST7701,width=800,height=480,to_ide=True)
 s=Sensor(id=2,width=1920,height=1080,fps=30)
 s.reset()
 for chn,fmt in ((CAM_CHN_ID_0,Sensor.RGB565),(CAM_CHN_ID_1,Sensor.RGB888),(CAM_CHN_ID_2,Sensor.RGBP888)):
  s.set_framesize(width=640,height=360,crop=False,chn=chn)
  s.set_pixformat(fmt,chn=chn)
 MediaManager.init()
 s.run()
 for i in range(20):
  for chn in (CAM_CHN_ID_0,CAM_CHN_ID_1,CAM_CHN_ID_2):s.snapshot(chn=chn)
 for chn in (CAM_CHN_ID_0,CAM_CHN_ID_1,CAM_CHN_ID_2):
  f=s.snapshot(chn=chn)
  with open(root+'/chn%d.raw'%chn,'wb') as out:out.write(f.bytearray())
  print('FORMAT_CHN',chn,f.format(),f.size(),str(f.to_numpy_ref().shape) if chn else '')
  if chn==0:
   f.save(root+'/chn0.bmp')
  elif chn==1:
   f.to_rgb565().save(root+'/chn1-converted.bmp')
  else:
   f.to_rgb565().save(root+'/chn2-converted.bmp')
 start=time.ticks_ms()
 while time.ticks_diff(time.ticks_ms(),start)<15000:
  os.exitpoint()
  f=s.snapshot(chn=CAM_CHN_ID_0)
  Display.show_image(f,x=80,y=60)
  r['frames']+=1
  f=None
  gc.collect()
 r['status']='COMPLETE'
except BaseException as e:
 r.update(status='FAILED',error=str(e))
 sys.print_exception(e)
finally:
 if s:s.stop()
 Display.deinit()
 MediaManager.deinit()
 with open(root+'/formats.json','w') as out:json.dump(r,out)
 print('FORMAT_DONE '+json.dumps(r))

