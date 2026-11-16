"""Bounded UVC model preview; USB evidence stays separate from CSI acceptance."""
import gc,json,time,os


def run(model_root,frames=600,report_path=None):
    if type(frames) is not int or not 5<=frames<=18000:raise ValueError('Bounded USB trial required')
    import image
    from media.display import Display
    from media.uvc import UVC
    from k230.topface_adapter import TopFaceBackend,annotate
    from k230.topface_service import resources
    backend=None;started=False;display=False;count=attempts=failures=consecutive=0
    last=None;wall_start=None;error=None;cleanup=[];actual=None;peak={};device=None;modes=[]
    timings={'capture':0,'jpeg_resize':0,'interleave':0,'infer':0,'display':0}
    try:
        connected,device=UVC.probe()
        if not connected:raise RuntimeError('USB camera is not enumerated')
        modes=[str(mode) for mode in UVC.list_video_mode()]
        ok,mode=UVC.select_video_mode(UVC.video_mode(1280,720,UVC.FOURCC_MJPEG,30))
        actual=str(mode)
        if not ok:raise RuntimeError('USB MJPEG negotiation failed')
        if mode.width!=1280 or mode.height!=720 or mode.fourcc!=UVC.FOURCC_MJPEG:
            raise ValueError('USB negotiated mode differs from preserved16:9 input')
        Display.init(Display.ST7701,width=800,height=480,to_ide=True);display=True
        canvas=image.Image(640,360,image.RGB565)
        backend=TopFaceBackend(model_root)
        UVC.start(cvt=False);started=True
        wall_start=time.ticks_ms()
        for index in range(frames):
            os.exitpoint();attempts+=1;t0=time.ticks_us()
            frame=UVC.snapshot(1000)
            if frame is None:
                failures+=1;consecutive+=1
                if consecutive>=3:raise RuntimeError('Three consecutive USB frames unavailable')
                continue
            consecutive=0;t1=time.ticks_us()
            packed=frame.to_rgb888(x_size=640,y_size=360)
            t2=time.ticks_us()
            array=packed.to_numpy_ref()
            if tuple(array.shape)!=(360,640,3):raise ValueError('USB prepared RGB HWC mismatch')
            planar=array.reshape((360*640,3)).transpose().copy().reshape((3,360,640))
            source=image.Image(640,360,image.RGBP888,alloc=image.ALLOC_REF,data=planar)
            t3=time.ticks_us();last=backend.infer(source);t4=time.ticks_us()
            # This only changes packed display storage, never the planar model copy.
            shown=image.Image(640,360,image.RGB888,alloc=image.ALLOC_REF,data=array,cvt_565=True)
            canvas.draw_image(shown,0,0);annotate(canvas,last)
            canvas.draw_string_advanced(8,335,18,'USB 1280x720 MJPEG',color=(255,255,255))
            Display.show_image(canvas,x=80,y=60);t5=time.ticks_us()
            for name,a,b in (('capture',t1,t0),('jpeg_resize',t2,t1),('interleave',t3,t2),
                              ('infer',t4,t3),('display',t5,t4)):
                timings[name]+=time.ticks_diff(a,b)/1000
            count+=1
            frame=packed=array=planar=source=shown=None
            gc.collect()
            if count%30==0:
                for name,row in resources().items():
                    if not isinstance(row,dict):continue
                    if row['used']>peak.get(name,{}).get('used',0):peak[name]=row
        status='COMPLETE_WITH_CAPTURE_LOSSES' if failures else 'COMPLETE'
    except BaseException as exc:
        status='FAILED';error=str(exc)
    finally:
        wall=time.ticks_diff(time.ticks_ms(),wall_start)/1000 if wall_start is not None else None
        frame=packed=array=planar=source=shown=canvas=None;gc.collect()
        if started:
            try:UVC.stop()
            except BaseException as exc:cleanup.append('uvc: '+str(exc))
        if display:
            try:Display.deinit()
            except BaseException as exc:cleanup.append('display: '+str(exc))
        if backend:
            try:backend.close()
            except BaseException as exc:cleanup.append('model: '+str(exc))
    result={'status':status,'error':error,'source':'USB_UVC','device':device,
            'actual_mode':actual,'modes':modes,'requested':frames,
            'attempted':attempts,'successful':count,'capture_failures':failures,'wall_seconds':wall,
            'fps':count/wall if wall else None,'mean_stage_ms':{k:v/count if count else None for k,v in timings.items()},
            'peak':peak,'cleanup_errors':cleanup,'last_observation':last,
            'input_transform':'MJPEG1280x720 ->RGB888640x360 -> ownedCHW; modelNCHW320x192',
            'calibration':'UNKNOWN_USB_CAMERA','model_sha256':backend.identity['kmodel_sha256'] if backend else None}
    if report_path:
        try:os.stat(report_path)
        except OSError:pass
        else:raise ValueError('Preserve previous USB receipt')
        with open(report_path,'w') as stream:json.dump(result,stream)
    print('USB_TOPFACE_RESULT '+json.dumps(result))
    return result
