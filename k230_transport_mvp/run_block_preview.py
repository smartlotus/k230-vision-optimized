"""User-assigned object/front cameras. Explicit CanMV entry; no motors."""
import sys


def main(max_frames=None, report_path=None, display_enabled=True, qr_enabled=True, confirmed_task=None, lock_exposure=False, diagnose_black=False,
         inspection_mode=None,inspection_target=1,inspection_station='RAW',pixel_profile=None,control_path=None,
         diagnostic_prefix=None,capture_path='/sdcard/transport_mvp/capture_request.json',save_samples=False,
         display_interval_ms=33,sensor_fps=30,max_elapsed_ms=None):
    if sys.implementation.name != 'micropython':
        raise SystemExit('Use CanMV on K230 to run block recognition.')
    import os, time, gc, json, image
    if type(display_interval_ms) is not int or not 1<=display_interval_ms<=1000:
        raise ValueError('Display interval must be integer1..1000ms')
    if type(sensor_fps) is not int or sensor_fps not in (30,60):
        raise ValueError('Primary sensor FPS request must be30 or60')
    if max_frames is not None and (type(max_frames) is not int or max_frames<1):
        raise ValueError('Frame limit must be a positive integer')
    if max_elapsed_ms is not None and (type(max_elapsed_ms) is not int or max_elapsed_ms<1):
        raise ValueError('Time limit must be a positive integer')
    if max_frames is not None and max_elapsed_ms is None:max_elapsed_ms=max(30000,max_frames*200)
    os.exitpoint(os.EXITPOINT_ENABLE)
    if '/sdcard/transport_mvp' not in sys.path:
        sys.path.insert(0,'/sdcard/transport_mvp')
    from media.sensor import Sensor
    from media.display import Display
    from media.media import MediaManager
    from vision.block_detector import BlockDetector
    from vision.task_preview import TaskPreview
    from k230.block_profile import parameters_for_camera, COLOR_NAMES, DRAW_COLORS, LIVE_UNVERIFIED_COLORS
    from k230.camera_roles import OBJECT_CAMERA_ID,FRONT_CAMERA_ID
    from k230.capture_io import CaptureIO
    from k230.preview_metrics import PreviewMetrics
    from k230.capture_health import CaptureHealth
    capture_io=CaptureIO()
    metrics=PreviewMetrics()
    capture_health=CaptureHealth()
    PARAMETERS=parameters_for_camera(OBJECT_CAMERA_ID)
    diagnostic_rows=[]
    def diagnose(label,im):
        path=diagnostic_prefix+'-'+label+'-live.jpg'
        if not capture_io.save_image(im,path):return
        points=[(im.width()//2,im.height()//2),(im.width()*3//4,im.height()//2)]
        diagnostic_rows.append({'stage':label,'path':path,'size':[im.width(),im.height()],
            'sample_pixels':[{'xy':point,'rgb':im.get_pixel(*point)} for point in points]})
        capture_io.save_json(diagnostic_rows,diagnostic_prefix+'-stages.json')
    from k230.exposure_control import configure_exposure
    inspector=None
    if inspection_mode is not None:
        from k230.visual_inspector import VisualInspector
        inspector=VisualInspector(inspection_mode,inspection_target,inspection_station,pixel_profile)
    detector=BlockDetector(PARAMETERS)
    task_preview=TaskPreview(qr_camera_id=FRONT_CAMERA_ID)
    if confirmed_task is not None:
        task_preview.restore_confirmed(confirmed_task)
    qr_frames=qr_errors=qr_total_us=0
    sensors={}
    media_ready=display_ready=False
    counts={2:0,1:0}
    errors={2:0,1:0}
    consecutive_errors={2:0,1:0}
    totals={'capture_us':0,'detect_us':0,'aux_us':0,'display_us':0,'capture_poll_us':0}
    capture_jobs=capture_job_total_us=capture_job_max_us=0
    pixel_frames=stable_pixel_frames=0
    def metric_values():
        value=dict(totals)
        value.update(frames=frames,display_frames=display_frames,pixel_frames=pixel_frames,
            stable_pixel_frames=stable_pixel_frames,capture_jobs=capture_jobs,capture_job_total_us=capture_job_total_us)
        for cid in (1,2):
            value['camera%d_frames' % cid]=counts[cid]
            value['camera%d_errors' % cid]=errors[cid]
        return value
    detect_times=[]
    frames=0
    display_frames=0
    latest=[]
    per_color={}
    visible_frames={str(c):0 for c in range(1,7)}
    stable_frames={str(c):0 for c in range(1,7)}
    all_six_visible=all_six_stable=0
    samples=[]
    changes=[]
    black_misses=[]
    last_signature=None
    started=None
    last_aux=last_display=last_log=last_qr=None
    last_control=None
    last_capture_poll=None
    inspection_result=None
    mode_started=None
    mode_frames=0
    per_mode_frames={}
    per_mode_detect_us={}
    per_mode_status={}
    aux=aux_full=None
    aux_ok=False
    aux_capture_ms=None
    fatal=None
    stop_reason=None
    exposure_state=None
    try:
        # Preserve the board's previously verified CSI2-first sensor initialization.
        # Processing roles are independent of physical initialization order.
        for cid in (2,1):
            sensor=None
            try:
                requested_fps=sensor_fps if cid==OBJECT_CAMERA_ID else 30
                sensor=Sensor(id=cid,width=1920,height=1080,fps=requested_fps)
                sensor.reset()
                sensor.set_framesize(width=320,height=240)
                sensor.set_pixformat(Sensor.RGB565)
                sensors[cid]=sensor
            except Exception as error:
                if 'IDE interrupt' in str(error): raise
                print('BLOCK_CAMERA_INIT_FAILED:'+json.dumps({'camera_id':cid,'error':str(error)}))
                if sensor is not None:
                    try:sensor.stop(is_del=True)
                    except Exception:pass
                if cid==OBJECT_CAMERA_ID:raise
        if display_enabled:
            Display.init(Display.ST7701,width=800,height=480,fps=30,to_ide=True)
            display_ready=True
        MediaManager.init();media_ready=True
        sensors[2].run()
        canvas=image.Image(800,480,image.RGB565) if display_enabled else None
        # Settle AE/AWB after sensor start; these frames are excluded from the benchmark.
        warmup=0;warmup_started=time.ticks_ms()
        while warmup<45 or time.ticks_diff(time.ticks_ms(),warmup_started)<2000:
            os.exitpoint()
            sensors[OBJECT_CAMERA_ID].snapshot(timeout=2000 if warmup==0 else 150,auto_reboot=False)
            warmup+=1
        exposure_state=configure_exposure(sensors[OBJECT_CAMERA_ID],lock_exposure)
        print('BLOCK_EXPOSURE:'+json.dumps(exposure_state))
        if exposure_state['locked']:
            for warmup in range(5):
                os.exitpoint();sensors[OBJECT_CAMERA_ID].snapshot(timeout=150,auto_reboot=False)
        started=time.ticks_ms()
        initial_context=(inspector.mode,inspector.target_id,inspector.station,inspector.seq) if inspector is not None else ('COLOR',None,'RAW',None)
        metrics.reset(initial_context,started,metric_values())
        print('BLOCK_STARTED:'+json.dumps({'primary':OBJECT_CAMERA_ID,'auxiliary':FRONT_CAMERA_ID,'quarantined':[0],
             'profile_id':PARAMETERS['profile_id'],'resolution':[320,240],'sensor_fps_request':sensor_fps,
             'auxiliary_fps_request':30,'sensor_input_size':[1920,1080],'motors_enabled':False}))
        while max_frames is None or frames<max_frames:
            os.exitpoint()
            now=time.ticks_ms()
            if max_elapsed_ms is not None and time.ticks_diff(now,started)>=max_elapsed_ms:
                stop_reason='TIME_LIMIT';break
            if inspector is not None and control_path and (last_control is None or time.ticks_diff(now,last_control)>=250):
                if inspector.poll_file(control_path):
                    aux_ok=False;aux_full=None;last_aux=None;mode_started=now;mode_frames=0
                    metrics.reset((inspector.mode,inspector.target_id,inspector.station,inspector.seq),now,metric_values())
                    print('VISION_MODE:'+json.dumps(inspector.status()))
                last_control=now
            primary_id=inspector.camera_id() if inspector is not None else OBJECT_CAMERA_ID
            auxiliary_id=1 if primary_id==2 else 2
            before=time.ticks_us()
            try:
                frame=sensors[primary_id].snapshot(timeout=2000 if counts[primary_id]==0 else 150,auto_reboot=False)
                if frame is None:raise RuntimeError('No primary camera frame')
                counts[primary_id]+=1
                consecutive_errors[primary_id]=0
                recovered=capture_health.success(primary_id,time.ticks_ms())
                if recovered:print('CAMERA_HEALTH:'+json.dumps(recovered))
            except Exception as error:
                if 'IDE interrupt' in str(error):raise
                errors[primary_id]+=1
                consecutive_errors[primary_id]+=1
                latest=[]
                inspection_result=None
                detector.reset()
                if inspector is not None:inspector.capture_failed()
                totals['capture_us']+=time.ticks_diff(time.ticks_us(),before)
                now=time.ticks_ms()
                event=capture_health.failure(primary_id,error,now)
                if event:print('CAMERA_HEALTH:'+json.dumps(event))
                # A failed primary must not starve the independent auxiliary view.
                if auxiliary_id in sensors and (last_aux is None or time.ticks_diff(now,last_aux)>=200):
                    aux_before=time.ticks_us()
                    try:
                        other=sensors[auxiliary_id].snapshot(timeout=80,auto_reboot=False)
                        if other is None:raise RuntimeError('No auxiliary frame')
                        aux=other.copy(x_scale=0.5,y_scale=0.5)
                        aux_ok=True;aux_capture_ms=time.ticks_ms();counts[auxiliary_id]+=1
                        recovered=capture_health.success(auxiliary_id,aux_capture_ms)
                        if recovered:print('CAMERA_HEALTH:'+json.dumps(recovered))
                    except Exception as aux_error:
                        if 'IDE interrupt' in str(aux_error):raise
                        errors[auxiliary_id]+=1;aux_ok=False
                        event=capture_health.failure(auxiliary_id,aux_error,time.ticks_ms())
                        if event:print('CAMERA_HEALTH:'+json.dumps(event))
                    last_aux=now;totals['aux_us']+=time.ticks_diff(time.ticks_us(),aux_before)
                if canvas and (last_display is None or time.ticks_diff(now,last_display)>=100):
                    canvas.clear()
                    canvas.draw_string_advanced(20,20,28,'CSI%d CAPTURE ERROR / retry' % primary_id,color=(255,70,70))
                    canvas.draw_string_advanced(20,65,18,str(error)[:55],color=(255,255,255))
                    if aux_ok and aux is not None and time.ticks_diff(time.ticks_ms(),aux_capture_ms)<400:
                        canvas.draw_image(aux,640,0)
                    else:
                        canvas.draw_string_advanced(642,35,16,'CSI%d OFFLINE' % auxiliary_id,color=(255,70,70))
                    canvas.draw_string_advanced(642,125,18,'CSI%d AUX' % auxiliary_id,color=(255,255,0))
                    Display.show_image(canvas)
                    last_display=now;display_frames+=1
                if last_log is None or time.ticks_diff(now,last_log)>=1000:
                    print('CAPTURE_FAULT_LIVE:'+json.dumps({'timestamp_ms':now,'frames':frames,'counts':counts,
                        'errors':errors,'camera_health':capture_health.rows,'observation':None,
                        'inspection':inspector.status() if inspector is not None else None,'motors_enabled':False}))
                    last_log=now
                time.sleep_ms(10)
                continue
            totals['capture_us']+=time.ticks_diff(time.ticks_us(),before)
            now=time.ticks_ms()
            frames+=1
            mode_frames+=1
            active_mode=inspector.mode if inspector is not None else 'COLOR'
            per_mode_frames[active_mode]=per_mode_frames.get(active_mode,0)+1
            if mode_started is None:mode_started=now
            if capture_path and (last_capture_poll is None or time.ticks_diff(now,last_capture_poll)>=500):
                before_poll=time.ticks_us()
                captured=capture_io.poll(capture_path,frame,{'camera_id':primary_id,'frame_id':frames,
                    'timestamp_ms':now,'image_profile':{'width':320,'height':240,'crop':False,
                    'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}},image.rgb_to_lab,os.exitpoint)
                poll_us=time.ticks_diff(time.ticks_us(),before_poll)
                totals['capture_poll_us']+=poll_us
                if captured:
                    capture_jobs+=1;capture_job_total_us+=poll_us
                    capture_job_max_us=max(capture_job_max_us,poll_us)
                    print('LIVE_CAPTURE:'+json.dumps(captured))
                last_capture_poll=now
            if diagnostic_prefix and frames==30:diagnose('primary-before-detect',frame)
            before=time.ticks_us()
            if inspector is None:
                latest=detector.process(frame,frames,now)
            else:
                packet={'image':frame,'camera_id':primary_id,'frame_id':frames,'timestamp_ms':now,
                    'width':320,'height':240,'is_simulated':False,
                    'image_profile':{'width':320,'height':240,'crop':False,'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}}
                inspection_result=inspector.process(packet)
                latest=inspector.adapter.last_block_candidates if active_mode=='COLOR' else []
                if active_mode=='TOP' and inspection_result and inspection_result.get('bbox') is not None:
                    latest=[{'target_id':inspector.target_id,'bbox':inspection_result['bbox'],
                        'center':(inspection_result['u_px'],inspection_result['v_px']),
                        'clipped':False,'overlapping':False,'stable':inspection_result['stable'],
                        'eligible':True,'track_id':0}]
                if active_mode=='QR' and inspection_result and inspection_result.get('raw_payload'):
                    task_preview.scan([inspection_result['raw_payload']],frames,now,camera_id=FRONT_CAMERA_ID)
            detect_us=time.ticks_diff(time.ticks_us(),before)
            totals['detect_us']+=detect_us
            if inspection_result and inspection_result.get('u_px') is not None:
                pixel_frames+=1
                stable_pixel_frames+=bool(inspection_result.get('stable'))
            if diagnostic_prefix and frames==30:diagnose('primary-after-detect',frame)
            per_mode_detect_us[active_mode]=per_mode_detect_us.get(active_mode,0)+detect_us
            status=inspection_result['status'] if inspection_result else ('PIXEL_CANDIDATES' if latest else 'NO_CANDIDATES')
            bucket=per_mode_status.setdefault(active_mode,{})
            bucket[status]=bucket.get(status,0)+1
            if len(detect_times)<2000:detect_times.append(detect_us)
            metric_latest=latest if active_mode=='COLOR' else []
            for target in metric_latest:
                cid=str(target['target_id'])
                per_color[cid]=per_color.get(cid,0)+1
            visible=set(t['target_id'] for t in metric_latest)
            stable=set(t['target_id'] for t in metric_latest if t['stable'])
            for cid in visible:visible_frames[str(cid)]+=1
            for cid in stable:stable_frames[str(cid)]+=1
            all_six_visible+=len(visible)==6
            all_six_stable+=len(stable)==6
            if diagnose_black and active_mode=='COLOR' and 5 not in visible and len(black_misses)<20:
                raw=frame.find_blobs([PARAMETERS['thresholds'][5]],roi=PARAMETERS['roi'],
                    pixels_threshold=20,area_threshold=50,merge=False,x_stride=2,y_stride=2)
                stat=frame.get_statistics(roi=(150,80,20,30))
                miss={'frame_id':frames,'timestamp_ms':now,
                    'raw_black_components':[{'bbox':b.rect(),'pixels':b.pixels(),'density':b.density()} for b in raw],
                    'assumed_black_interior_roi':[150,80,20,30],
                    'roi_lab_median':[stat.l_median(),stat.a_median(),stat.b_median()]}
                if len(black_misses)<3:
                    path='/sdcard/transport_mvp/black-miss-'+str(frames)+'-live.jpg'
                    if capture_io.save_image(frame,path):miss['image_path']=path
                black_misses.append(miss)
            signature=tuple((cid,sum(t['target_id']==cid for t in latest),
                                 sum(t['target_id']==cid and t['stable'] for t in latest)) for cid in range(1,7))
            if signature!=last_signature:
                change={'frame_id':frames,'timestamp_ms':now,'colors_counts_stable':signature}
                changes.append(change)
                changes=changes[-200:]
                print('BLOCK_CHANGE:'+json.dumps(change))
                last_signature=signature
            if save_samples and frames in (30,90,150,300):
                path='/sdcard/transport_mvp/block-eval-'+str(frames)+'-live.jpg'
                if capture_io.save_image(frame,path):samples.append({'frame_id':frames,'path':path,'detections':latest})
            # Auxiliary view and display are rate-limited independently of detection.
            if auxiliary_id in sensors and (last_aux is None or time.ticks_diff(now,last_aux)>=200 or (diagnostic_prefix and frames==30)):
                before=time.ticks_us()
                try:
                    other=sensors[auxiliary_id].snapshot(timeout=2000 if counts[auxiliary_id]==0 else 80,auto_reboot=False)
                    if other is None:raise RuntimeError('No CSI%d frame' % auxiliary_id)
                    if inspector is None and qr_enabled and task_preview.task is None and (last_qr is None or time.ticks_diff(now,last_qr)>=300):
                        qr_start=time.ticks_us()
                        try:
                            payloads=[code.payload() for code in other.to_grayscale().find_qrcodes()]
                            qr_frames+=1
                            update=task_preview.scan(payloads,qr_frames,time.ticks_ms(),camera_id=FRONT_CAMERA_ID)
                            if update['submitted']:
                                print('TASK_QR_CONFIRMED:'+json.dumps(task_preview.summary()))
                            elif payloads:
                                print('TASK_QR_SCAN:'+json.dumps(update))
                        except Exception as error:
                            if 'IDE interrupt' in str(error):raise
                            qr_errors+=1
                            print('TASK_QR_ERROR:'+str(error))
                        qr_total_us+=time.ticks_diff(time.ticks_us(),qr_start)
                        last_qr=now
                    aux=other.copy(x_scale=0.5,y_scale=0.5)
                    if diagnostic_prefix and frames==30:
                        diagnose('primary-after-aux',frame);diagnose('aux-raw',other);diagnose('aux-copy',aux)
                    aux_full=other.copy() if inspector is None and qr_enabled and task_preview.task is None else None
                    aux_ok=True
                    aux_capture_ms=now
                    counts[auxiliary_id]+=1
                    recovered=capture_health.success(auxiliary_id,time.ticks_ms())
                    if recovered:print('CAMERA_HEALTH:'+json.dumps(recovered))
                except Exception as error:
                    if 'IDE interrupt' in str(error):raise
                    errors[auxiliary_id]+=1
                    aux_ok=False
                    event=capture_health.failure(auxiliary_id,error,time.ticks_ms())
                    if event:print('CAMERA_HEALTH:'+json.dumps(event))
                last_aux=now
                totals['aux_us']+=time.ticks_diff(time.ticks_us(),before)
            if canvas and (last_display is None or time.ticks_diff(now,last_display)>=display_interval_ms or (diagnostic_prefix and frames==30)):
                before=time.ticks_us()
                canvas.clear()
                canvas.draw_image(frame,0,0,x_scale=2,y_scale=2)
                if diagnostic_prefix and frames==30:diagnose('canvas-primary',canvas)
                if inspector is not None:
                    canvas.draw_string_advanced(8,8,20,'CSI%d %s' % (primary_id,active_mode),color=(255,255,0))
                roi=PARAMETERS['roi']
                if inspector is not None and active_mode!='COLOR':
                    if active_mode=='LANE':roi=inspector.config.get('LANE_ROI') or (0,0,320,240)
                    elif active_mode in ('RING','TOP'):
                        cfg=inspector.config.get(active_mode+'_CONFIG',{}).get(inspector.station,{})
                        roi=cfg.get('roi') or (0,0,320,240)
                    else:roi=(0,0,320,240)
                rx,ry,rw,rh=roi
                canvas.draw_rectangle(rx*2,ry*2,rw*2,rh*2,color=(180,180,180),thickness=1)
                canvas.draw_string_advanced(4,max(0,ry*2-18),14,'OBSERVATION WINDOW',color=(255,255,255))
                for target in latest:
                    x,y,w,h=target['bbox']
                    color=DRAW_COLORS[target['target_id']]
                    canvas.draw_rectangle(int(x*2),int(y*2),int(w*2),int(h*2),color=color,thickness=2)
                    u,v=target['center']
                    canvas.draw_cross(int(u*2),int(v*2),color=color,size=9)
                    label='%s %s' % (COLOR_NAMES[target['target_id']],
                        'C' if target['clipped'] else ('O' if target['overlapping'] else ('S' if target['stable'] else '?')))
                    canvas.draw_string_advanced(int(x*2),int(max(0,y*2-18)),14,label,color=color)
                selected=task_preview.selected_target(latest) if inspector is None else next((c for c in latest if c['target_id']==inspector.target_id and c['eligible'] and c['stable']),None)
                if selected:
                    x,y,w,h=selected['bbox']
                    canvas.draw_rectangle(int(x*2),int(y*2),int(w*2),int(h*2),color=(255,255,255),thickness=4)
                    canvas.draw_string_advanced(int(x*2),int(y*2+20),18,'NEXT (PX)',color=(255,255,255))
                summary=task_preview.summary()
                if inspector is not None:
                    state=inspection_result or {'status':'IDLE','reason':None}
                    canvas.draw_string_advanced(8,392,18,'%s target=%s station=%s' % (active_mode,inspector.target_id,inspector.station),color=(255,255,255))
                    canvas.draw_string_advanced(8,416,16,state['status'],color=(255,255,0))
                    canvas.draw_string_advanced(8,438,14,str(state.get('reason') or 'PX ONLY / NO H'),color=(255,255,255))
                    for candidate in state.get('ring_candidates',[]):
                        u,v=candidate['center']
                        canvas.draw_cross(int(u*2),int(v*2),size=8,color=(255,80,255))
                    if state.get('u_px') is not None:
                        canvas.draw_cross(int(state['u_px']*2),int(state['v_px']*2),size=10,color=(255,255,255))
                elif summary['payload']:
                    canvas.draw_string_advanced(8,392,18,'TASK '+summary['payload'],color=(255,255,255))
                    for row,key in enumerate(('batch1_colors','batch2_colors')):
                        order=' > '.join(COLOR_NAMES[c] for c in summary[key])
                        canvas.draw_string_advanced(8,414+row*22,16,'B%d: %s' % (row+1,order),color=(255,255,0))
                else:
                    canvas.draw_string_advanced(8,420,18,'CSI%d: show task QR / waiting' % FRONT_CAMERA_ID,color=(255,255,0))
                canvas.draw_string_advanced(8,461,14,'CSI%d 320x240 PX / no physical motion' % primary_id,color=(255,255,255))
                if aux_ok and aux is not None and time.ticks_diff(now,aux_capture_ms)<400:
                    canvas.draw_image(aux,640,0)
                else:
                    canvas.draw_string_advanced(642,35,16,'CSI%d OFFLINE' % auxiliary_id,color=(255,70,70))
                canvas.draw_string_advanced(642,125,18,'CSI%d AUX' % auxiliary_id,color=(255,255,0))
                elapsed=max(1,time.ticks_diff(now,started))
                view_elapsed=max(1,time.ticks_diff(now,mode_started))
                recent=metrics.windows[-1] if metrics.windows and metrics.windows[-1]['context']==metrics.context else None
                rate='%.1f det/s' % recent['processing_fps'] if recent else 'Measuring FPS'
                alg_ms=recent['delta']['detect_us']/max(1,recent['delta']['frames'])/1000 if recent else None
                canvas.draw_string_advanced(642,155,18,rate,color=(0,255,80))
                canvas.draw_string_advanced(642,183,16,'%.1f ms alg' % alg_ms if alg_ms is not None else 'Waiting1s',color=(255,255,255))
                canvas.draw_string_advanced(642,211,18,'CSI0 FLAGGED',color=(255,70,70))
                canvas.draw_string_advanced(642,239,14,'PX' if PARAMETERS['sampling_verified'] else 'PROFILE UNVERIFIED',color=(255,255,0))
                for index,cid in enumerate(range(1,7)):
                    usable=[t for t in latest if t['target_id']==cid and t['stable'] and t['eligible']]
                    if usable:
                        target=min(usable,key=lambda t:(t['center'][0]-160)**2+(t['center'][1]-120)**2)
                        u,v=target['center']
                        text='%d %s %d,%d' % (cid,COLOR_NAMES[cid],u,v)
                    else:
                        text='%d %s WAIT' % (cid,COLOR_NAMES[cid])
                    canvas.draw_string_advanced(642,273+index*23,14,text,color=DRAW_COLORS[cid])
                canvas.draw_string_advanced(642,418,14,'S=stable C=clip',color=(255,255,255))
                canvas.draw_string_advanced(642,440,14,'O=overlap ?=wait',color=(255,255,255))
                if inspector is None and qr_enabled and task_preview.task is None and aux_full is not None and aux_ok:
                    # During setup the scanning camera is large, so users target the right lens.
                    canvas.draw_image(aux_full,0,0,x_scale=2,y_scale=2)
                    canvas.draw_image(frame,640,0,x_scale=0.5,y_scale=0.5)
                    canvas.draw_rectangle(120,40,400,400,color=(0,255,255),thickness=2)
                    canvas.draw_string_advanced(8,8,24,'CSI%d SCAN QR' % FRONT_CAMERA_ID,color=(0,255,255))
                    canvas.draw_string_advanced(8,442,16,'Show QR here / CSI%d blocks still running' % OBJECT_CAMERA_ID,color=(255,255,0))
                    canvas.draw_rectangle(640,122,160,30,color=(0,0,0),fill=True)
                    canvas.draw_string_advanced(642,125,18,'CSI%d BLOCKS' % OBJECT_CAMERA_ID,color=(255,255,0))
                Display.show_image(canvas)
                display_frames+=1
                if diagnostic_prefix and frames==30:diagnose('canvas-composite',canvas)
                last_display=now
                totals['display_us']+=time.ticks_diff(time.ticks_us(),before)
            end_ms=time.ticks_ms()
            context=(active_mode,inspector.target_id,inspector.station,inspector.seq) if inspector is not None else ('COLOR',None,'RAW',None)
            window=metrics.update(context,end_ms,metric_values())
            if window is not None:print('PREVIEW_WINDOW:'+json.dumps(window))
            if last_log is None or time.ticks_diff(end_ms,last_log)>=1000:
                now=end_ms
                elapsed=max(1,time.ticks_diff(now,started))
                print('BLOCK_LIVE:'+json.dumps({'frames':frames,'elapsed_ms':elapsed,'detections_per_second':frames*1000/elapsed,
                    'display_frames':display_frames,'display_frames_per_second':display_frames*1000/elapsed,
                    'display_interval_ms':display_interval_ms,
                    'counts':counts,'errors':errors,'targets':latest,'task':task_preview.summary(),
                    'inspection':inspector.status() if inspector is not None else None,'observation':inspection_result,
                    'qr_frames':qr_frames,'qr_errors':qr_errors,'mean_detect_us':totals['detect_us']/frames,
                    'capture_io':capture_io.status(),'motors_enabled':False}))
                last_log=now
                gc.collect()
    except KeyboardInterrupt:
        stop_reason='USER_STOP'
        print('BLOCK_STOPPED')
    except Exception as error:
        if 'IDE interrupt' not in str(error):
            fatal=str(error)
            stop_reason='ERROR'
            sys.print_exception(error)
        else:stop_reason='USER_STOP'
    finally:
        metrics.finish(time.ticks_ms(),metric_values())
        elapsed=time.ticks_diff(time.ticks_ms(),started) if started is not None else 0
        for sensor in sensors.values():
            try:sensor.stop()
            except Exception as error:print('block camera cleanup',error)
        if display_ready:
            try:Display.deinit()
            except Exception as error:print('block display cleanup',error)
        if media_ready:
            try:MediaManager.deinit()
            except Exception as error:print('block media cleanup',error)
        ordered=sorted(detect_times)
        report={'test':'LIVE_BLOCK_RECOGNITION','source':'REAL_CAMERA','profile_id':PARAMETERS['profile_id'],
                'stop_reason':stop_reason or 'SAMPLE_LIMIT','requested_frames':max_frames,
                'max_elapsed_ms':max_elapsed_ms,'completed_requested_frames':max_frames is not None and frames>=max_frames,
                'camera_health':capture_health.rows,
                'sensor_fps_request':sensor_fps,'auxiliary_fps_request':30,'sensor_input_size':[1920,1080],
                'performance_windows':metrics.windows,'discarded_partial_windows':metrics.discarded_partial_windows,
                'window_covered_frames':sum(w['delta']['frames'] for w in metrics.windows),
                'discarded_partial_frames':metrics.discarded_partial_frames,
                'pixel_frames':pixel_frames,'stable_pixel_frames':stable_pixel_frames,
                'capture_jobs':capture_jobs,'capture_job_total_us':capture_job_total_us,'capture_job_max_us':capture_job_max_us,
                'display_frames':display_frames,'display_interval_ms':display_interval_ms,
                'display_frames_per_second':display_frames*1000/elapsed if elapsed else 0,
                'capture_io':capture_io.status(),
                'object_camera_id':OBJECT_CAMERA_ID,'front_camera_id':FRONT_CAMERA_ID,
                'profile_measurement_camera_id':PARAMETERS['measurement_camera_id'],
                'profile_sampling_verified':PARAMETERS['sampling_verified'],
                'detection_roi':PARAMETERS['roi'],'exposure_state':exposure_state,
                'frames':frames,'elapsed_ms':elapsed,'detections_per_second':frames*1000/elapsed if elapsed else 0,
                'counts':counts,'errors':errors,'stage_total_us':totals,'per_color_detection_count':per_color,
                'per_color_visible_frames':visible_frames,'per_color_stable_frames':stable_frames,
                'all_six_visible_frames':all_six_visible,'all_six_stable_frames':all_six_stable,
                'mean_detect_us':totals['detect_us']/frames if frames else None,
                'p95_detect_us':ordered[int((len(ordered)-1)*0.95)] if ordered else None,
                'samples':samples,'changes':changes,'latest':latest,'task':task_preview.summary(),
                'inspection':inspector.status() if inspector is not None else None,'observation':inspection_result,'per_mode_frames':per_mode_frames,
                'per_mode_detect_us':per_mode_detect_us,'per_mode_status':per_mode_status,
                'color_metric_frames':per_mode_frames.get('COLOR',0),
                'color_metrics_scope':'COLOR mode only; TOP counts excluded; presence is not accuracy',
                'black_miss_diagnostics':black_misses,
                'qr_frames':qr_frames,'qr_errors':qr_errors,'qr_total_us':qr_total_us,'fatal':fatal,'motors_enabled':False,
                'metric_coordinates_available':False,
                'unsampled_colors':LIVE_UNVERIFIED_COLORS if PARAMETERS['sampling_verified'] else tuple(range(1,7))}
        if report_path:
            capture_io.save_json(report,report_path)
        print('BLOCK_RESULT:'+json.dumps(report))
    return report


if __name__=='__main__':main()
