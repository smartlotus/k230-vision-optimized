"""Original camera/Observation adapters with the accepted TopFace TOP backend."""
import gc,json,time,os
from vision.common import observation
from .camera_manager import CameraManager
from .vision_adapter import VisionAdapter
from .topface_adapter import TopFaceBackend,TopFacePreview,annotate


def resources():
    result={'python_heap_used':gc.mem_alloc(),'python_heap_free':gc.mem_free()}
    for name in ('sys_heap','sys_page','sys_mmz'):
        try:
            total,used,free=getattr(gc,name)()
            result[name]={'total':total,'used':used,'free':free}
        except (AttributeError,ValueError,OSError):pass
    return result


def configuration(model_root,overrides=None):
    from . import config as defaults
    cfg={key:getattr(defaults,key) for key in dir(defaults) if key.isupper()}
    cfg.update(WIDTH=640,HEIGHT=360,CAMERA_IDS=(2,1),PREVIEW=False,
               TOPFACE_MODEL_ROOT=model_root,UART_ENABLED=False)
    cfg['TOP_CONFIG']={'BUFFER':{'roi':[0,0,640,360],'reference':[320,180],
                               'min_area':1,'max_area':640*360}}
    if overrides:cfg.update(overrides)
    if (cfg['WIDTH'],cfg['HEIGHT'])!=(640,360) or cfg.get('CROP') or cfg.get('ROTATION',0):
        raise ValueError('Frozen TopFace requires uncropped 640x360, rotation0')
    if cfg.get('HMIRROR') or cfg.get('VFLIP'):raise ValueError('Frozen model camera orientation differs')
    if cfg.get('UART_ENABLED') or cfg.get('UART',{}).get('enabled'):raise ValueError('UART remains disabled')
    return cfg


def run(model_root,frames=1000,mode='TOP',target_id=1,station='BUFFER',
        display=True,ide_preview=True,report_path=None,config_override=None,gc_interval=4):
    if type(frames) is not int or not 5<=frames<=18000:raise ValueError('Bounded 5..18000 frames required')
    if mode not in ('TOP','COLOR','RING','QR','LANE'):raise ValueError('Unsupported mode')
    if type(gc_interval) is not int or not 1<=gc_interval<=4:raise ValueError('GC interval1..4 required')
    from media.display import Display
    import image
    cfg=configuration(model_root,config_override)
    backend=vision=camera=None;preview=TopFacePreview();canvas=None
    counts={'capture':[],'inference':[],'display':[],'gc':[],'cycle':[],
            'ai2d':[],'kpu':[],'decode':[]}
    attempts=successful=failures=consecutive=displayed=0;errors=[];last=None
    state='STARTED';error_text=None;cleanup=[];started=None
    peak={};initial=resources()
    try:
        backend=TopFaceBackend(model_root);vision=VisionAdapter(cfg,topface_backend=backend)
        if display:
            Display.init(Display.ST7701,width=800,height=480,to_ide=ide_preview)
            canvas=image.Image(640,360,image.RGB565)
        camera=CameraManager(cfg);camera.start()
        for _ in range(5):
            packet=camera.get_frame(camera.camera_for_mode(mode),mode=mode)
            packet=None;gc.collect()
        started=time.ticks_ms()
        for index in range(frames):
            os.exitpoint();attempts+=1;before=time.ticks_us()
            try:
                packet=camera.get_frame(camera.camera_for_mode(mode),mode=mode)
                if packet is None:raise RuntimeError('NO_FRAME')
            except RuntimeError as error:
                failures+=1;consecutive+=1
                if len(errors)<8:errors.append({'attempt':attempts,'error':str(error)})
                if consecutive>=3:raise
                time.sleep_ms(100);continue
            captured=time.ticks_us();consecutive=0
            last=vision.detect(mode,packet,target_id,station)
            if last['status']=='ERROR':raise RuntimeError(last['reason'])
            if mode=='TOP':
                for key in ('ai2d','kpu','decode'):counts[key].append(backend.timing[key+'_ms'])
            inferred=time.ticks_us()
            if display:
                frame=packet['image']
                shown=preview.convert(frame) if mode=='TOP' else frame
                canvas.draw_image(shown,0,0)
                if mode=='TOP':annotate(canvas,vision.last_topface)
                else:canvas.draw_string_advanced(8,8,20,mode+' '+last['status'],color=(255,255,255))
                Display.show_image(canvas,x=80,y=60);displayed+=1
                shown=frame=None;preview.clear()
            rendered=time.ticks_us()
            packet=None
            if (index+1)%gc_interval==0 or gc.mem_free()<1000000:gc.collect()
            collected=time.ticks_us()
            for key,end,begin in (('capture',captured,before),('inference',inferred,captured),
                                  ('display',rendered,inferred),('gc',collected,rendered),
                                  ('cycle',collected,before)):
                counts[key].append(time.ticks_diff(end,begin)/1000)
            successful+=1
            if successful%30==0:
                sample=resources()
                for name,row in sample.items():
                    used=row['used'] if isinstance(row,dict) else row
                    if name.endswith('free'):continue
                    if used>peak.get(name,{}).get('used',-1):
                        peak[name]=dict(row) if isinstance(row,dict) else {'used':used}
        state='RUN_COMPLETE_WITH_CAPTURE_LOSSES' if failures else 'RUN_COMPLETE'
    except BaseException as error:
        state='RUN_FAILED';error_text=str(error)
    finally:
        wall=time.ticks_diff(time.ticks_ms(),started)/1000 if started is not None else None
        packet=shown=frame=canvas=None;preview.clear()
        if camera:
            try:camera.close();cleanup.extend(camera.close_errors)
            except BaseException as error:cleanup.append('camera: '+str(error))
        if display:
            try:Display.deinit()
            except BaseException as error:cleanup.append('display: '+str(error))
        if backend:
            try:backend.close()
            except BaseException as error:cleanup.append('model: '+str(error))
        gc.collect()
    def stats(values):
        values=sorted(values);n=len(values)
        return {'n':n,'mean_ms':sum(values)/n if n else None,
                'p50_ms':values[(n-1)//2] if n else None,
                'p95_ms':values[min(n-1,(n*95+99)//100-1)] if n else None}
    report={'status':state,'error':error_text,'requested':frames,'attempted':attempts,
            'successful':successful,'capture_failures':failures,'errors_first_eight':errors,
            'displayed':displayed,'wall_seconds':wall,'effective_fps':successful/wall if wall else None,
            'timing':{key:stats(values) for key,values in counts.items()},'last_observation':last,
            'model_sha256':backend.identity['kmodel_sha256'] if backend else None,
            'cleanup_errors':cleanup,'mode':mode,'target_id':target_id,'camera_ids':[2,1],
            'resource_initial':initial,'resource_peak_sampled':peak,'resource_after_cleanup':resources(),
            'gc_interval':gc_interval,'gc_heap_free_trigger_bytes':1000000,
            'cpu_utilization_percent':None,'kpu_hardware_utilization_percent':None,
            'uart_enabled':False,'motors_enabled':False,'profile_saved':False}
    if report_path:
        try:os.stat(report_path)
        except OSError:pass
        else:raise ValueError('Preserve existing report; choose fresh path')
        with open(report_path,'w') as stream:json.dump(report,stream)
    print('TOPFACE_SERVICE_RESULT '+json.dumps(report))
    return report
