import os,sys,time,gc,json,hashlib,binascii
import image
from media.sensor import Sensor,CAM_CHN_ID_0
from media.display import Display
from media.media import MediaManager
from nonai2d import CSC
root='/sdcard/cdiag1116'
s=None;c=None
r={'kind':'same_frame_conversion','files':[]}
def digest(data):
 return binascii.hexlify(hashlib.sha256(data).digest()).decode()
try:
 Display.init(Display.ST7701,width=800,height=480,to_ide=True)
 s=Sensor(id=2,width=1920,height=1080,fps=30)
 s.reset()
 s.set_framesize(width=640,height=360,crop=False,chn=CAM_CHN_ID_0)
 s.set_pixformat(Sensor.RGBP888,chn=CAM_CHN_ID_0)
 MediaManager.init()
 s.run()
 for i in range(25):s.snapshot(chn=CAM_CHN_ID_0)
 f=s.snapshot(chn=CAM_CHN_ID_0)
 b=bytes(f.bytearray())
 with open(root+'/same-source.raw','wb') as out:out.write(b)
 r['before_sha']=digest(b);r['before_format']=f.format()
 native=f.to_rgb565()
 native.save(root+'/same-native.bmp')
 r['after_native_sha']=digest(bytes(f.bytearray()))
 r['after_native_format']=f.format()
 c=CSC(CSC.PIXEL_FORMAT_RGB_565)
 converted=c.convert(f)
 converted.save(root+'/same-csc.bmp')
 r['after_csc_sha']=digest(bytes(f.bytearray()))
 r['source_unchanged']=r['before_sha']==r['after_native_sha']==r['after_csc_sha']
 r['status']='CONVERSIONS_COMPLETE'
 print('CONVERSION_DIAG '+json.dumps(r))
 start=time.ticks_ms()
 while time.ticks_diff(time.ticks_ms(),start)<15000:
  os.exitpoint()
  Display.show_image(converted,x=80,y=60)
  time.sleep_ms(100)
except BaseException as e:
 r.update(status='FAILED',error=str(e))
 sys.print_exception(e)
finally:
 if s is not None:s.stop()
 Display.deinit()
 if c is not None:c.destroy()
 MediaManager.deinit()
 gc.collect()
 with open(root+'/same-conversion.json','w') as out:json.dump(r,out)
 print('CONVERSION_DONE '+json.dumps(r))

