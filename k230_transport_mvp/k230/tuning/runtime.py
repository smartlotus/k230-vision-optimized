"""LCD/touch owner. Cameras stay OFF until the operator explicitly enables them."""
import sys
from .schema import image_profile
from .store import ProfileStore
from .model import TuningSession
from .panel import Panel
from k230.camera_roles import MODE_CAMERA


class TuningApp:
    def __init__(self,root='/sdcard/transport_mvp/tuning_profiles',demo=False):
        self.session=TuningSession(ProfileStore(root),demo=demo)
        self.panel=Panel(self.session);self.sensors={};self.errors={}
        self.display_ready=False;self.media_ready=False;self.canvas=None
        self.touches=[];self.frame=None;self.frozen_frame=None
        self.signature=None;self.frame_id=0;self.last_capture=None;self.last_draw=None
        self.last_status='摄像头未开启';self.capture_errors={};self.applied_geometry=None
        self.cleanup_errors=[]
        self.frame_metadata=None;self.last_sample=None

    def close_display(self):
        self.frame=None;self.frozen_frame=None;self.canvas=None;self.frame_metadata=None
        for sensor in self.sensors.values():
            try:sensor.stop()
            except Exception:pass
        self.sensors={}
        if self.display_ready:
            try:self.Display.deinit()
            except Exception as error:self.cleanup_errors.append('display: '+str(error))
            finally:self.display_ready=False
        if self.media_ready:
            try:self.MediaManager.deinit()
            except Exception as error:self.cleanup_errors.append('media: '+str(error))
            finally:self.media_ready=False

    def open_display(self,live,to_ide=True):
        import image
        from media.display import Display
        from media.media import MediaManager
        self.Display=Display;self.MediaManager=MediaManager
        self.close_display();self.errors={}
        p=self.session.active;b=p['binding'];cam=p['camera']
        if live:
            from media.sensor import Sensor
            for cid in (2,1):
                sensor=None
                try:
                    sensor=Sensor(id=cid,width=b['input_width'],height=b['input_height'],fps=cam['fps'])
                    sensor.reset();sensor.set_framesize(width=b['width'],height=b['height'],crop=False)
                    sensor.set_pixformat(Sensor.RGB565);sensor.set_hmirror(b['hmirror']);sensor.set_vflip(b['vflip'])
                    self.sensors[cid]=sensor
                except Exception as error:
                    self.errors[cid]=str(error)
                    if sensor is not None:
                        try:sensor.stop(is_del=True)
                        except Exception:pass
        self.display_ready=True
        Display.init(Display.ST7701,width=800,height=480,fps=30,to_ide=to_ide)
        if self.sensors:
            self.media_ready=True;MediaManager.init()
            self.sensors[next(cid for cid in (2,1) if cid in self.sensors)].run()
            for cid,sensor in self.sensors.items():
                try:
                    sensor.auto_exposure(cam['auto_exposure'])
                    if not cam['auto_exposure']:sensor.exposure(cam['exposure_us'])
                except Exception as error:
                    self.errors[cid]='曝光设置失败：'+str(error)
                    self.panel.message='CSI%d 曝光设置未生效，请重新核对' % cid
        self.canvas=image.Image(800,480,image.RGB565)
        self.applied_geometry=(str(b),str(cam),live)
        self.frame=None;self.last_capture=None;self.last_draw=None
        self.last_status='摄像头启动中' if live else '摄像头未开启'

    def setup(self):
        import os
        from machine import TOUCH
        os.exitpoint(os.EXITPOINT_ENABLE)
        self.open_display(False)
        self.TOUCH=TOUCH
        for device in (0,getattr(TOUCH,'DEV_IDE',-1)):
            if device<0:continue
            try:
                touch=TOUCH(device,range_x=800,range_y=480) if device==getattr(TOUCH,'DEV_IDE',-1) else TOUCH(device)
                self.touches.append(touch);self.panel.touch_sources.append(device)
            except Exception as error:print('TUNING_TOUCH_UNAVAILABLE:'+str(device)+':'+str(error))
        if not self.touches:self.panel.message='触摸设备不可用：只读界面'

    def tick(self):
        import time,os
        os.exitpoint();panel=self.panel;s=self.session
        events={self.TOUCH.EVENT_DOWN:'down',self.TOUCH.EVENT_UP:'up',self.TOUCH.EVENT_MOVE:'move'}
        for source,touch in enumerate(self.touches):
            for point in touch.read(5):
                event=events.get(point.event)
                if event and 0<=point.x<800 and 0<=point.y<480:panel.handle_touch(event,point.x,point.y,(source,point.track_id))
        binding=s.active['binding'];cam=s.active['camera']
        live=panel.live_requested and not s.demo
        desired=(str(binding),str(cam),live)
        if desired!=self.applied_geometry:
            self.open_display(live);panel.frozen=False;s.observation=None;s.adapter.reset_session()
        cid=MODE_CAMERA[s.mode];signature=(cid,str(binding),s.apply_serial)
        if signature!=self.signature:
            self.frame=None;self.frozen_frame=None;panel.frozen=False;s.observation=None
            self.signature=signature
        now=time.ticks_ms()
        if panel.frozen:
            if self.frozen_frame is None and self.frame is not None:self.frozen_frame=self.frame.copy()
            s.observation=None;self.last_status='已冻结：暂停检测'
        else:
            self.frozen_frame=None
            if live and (self.last_capture is None or time.ticks_diff(now,self.last_capture)>=33):
                self.last_capture=now
                try:
                    if cid not in self.sensors:raise RuntimeError(self.errors.get(cid,'Camera unavailable'))
                    frame=self.sensors[cid].snapshot(timeout=100,auto_reboot=False)
                    if frame is None:raise RuntimeError('Empty frame')
                    if (frame.width(),frame.height())!=(binding['width'],binding['height']):raise RuntimeError('Image size differs from saved binding')
                    self.frame=frame;self.frame_id+=1;self.last_status='CSI%d 原始画面 / PX' % cid
                    self.frame_metadata={'camera_id':cid,'frame_id':self.frame_id,'timestamp_ms':time.ticks_ms(),
                        'is_simulated':False,'image_profile':image_profile(s.active),'mode':s.mode,'station':s.station,
                        'profile_revision':s.active['revision']}
                    # QR/LANE may convert their input to grayscale in place.
                    detection_frame=frame.copy() if s.mode in ('QR','LANE') else frame
                    s.process({'image':detection_frame,'camera_id':cid,'frame_id':self.frame_id,'timestamp_ms':time.ticks_ms(),
                        'is_simulated':False,'image_profile':image_profile(s.active)})
                except Exception as error:
                    if 'IDE interrupt' in str(error):raise
                    self.frame=None;s.observation=None;s.adapter.reset_session()
                    self.frame_metadata=None
                    self.capture_errors[cid]=str(error);self.last_status='CSI%d 取帧失败；菜单仍可操作' % cid
            elif not live:self.frame=None;s.observation=None
        if panel.capture_requested:
            panel.capture_requested=False
            try:
                if self.frame_metadata is None or self.frame is None:raise ValueError('No current native frame')
                if self.frame_metadata['mode']!=s.mode or self.frame_metadata['station']!=s.station:
                    raise ValueError('Wait for a fresh frame in the selected context')
                import image
                from .sampling import SampleWriter
                frozen=(self.frozen_frame if panel.frozen else self.frame).copy()
                self.last_sample=SampleWriter().save(frozen,self.frame_metadata,image.rgb_to_lab,os.exitpoint)
                print('TUNING_NATIVE_SAMPLE:'+__import__('json').dumps(self.last_sample))
                panel.message='样本%d已保存，预览继续' % self.last_sample['serial'] if self.last_sample['complete'] else '样本写入不完整，旧参数未变'
            except (OSError,ValueError) as error:panel.message='采样失败：'+str(error)[:20]
        if self.last_draw is None or time.ticks_diff(now,self.last_draw)>=100:
            panel.render(self.canvas,self.frozen_frame if panel.frozen else self.frame,self.last_status)
            self.Display.show_image(self.canvas);self.last_draw=now
        time.sleep_ms(5)

    def close(self):
        for touch in self.touches:
            try:touch.deinit()
            except Exception:pass
        self.touches=[]
        self.close_display()


def main(demo=False,root='/sdcard/transport_mvp/tuning_profiles',max_ticks=None):
    if sys.implementation.name!='micropython':raise SystemExit('Use CanMV on K230 for the touch panel.')
    app=TuningApp(root,demo);ticks=0
    try:
        app.setup()
        print('TUNING_PANEL_STARTED:'+str({'demo':demo,'touch_sources':app.panel.touch_sources,'camera_enabled':False}))
        while max_ticks is None or ticks<max_ticks:app.tick();ticks+=1
    except KeyboardInterrupt:print('TUNING_PANEL_STOPPED')
    finally:app.close()
    return app
