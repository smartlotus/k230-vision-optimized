"""Bounded CSI2, AI2D, KPU and display benchmark for CanMV."""
import gc
import json
import math
import time

from topface_decode import decode


def summarize(samples):
    values = sorted(samples)
    length = len(values)
    if not length:
        return {'frames': 0}
    def percentile(p):
        return values[min(length-1, (length*p+99)//100-1)]/1000
    return {'frames': length, 'p50_ms': percentile(50), 'p95_ms': percentile(95),
            'p99_ms': percentile(99), 'mean_ms': sum(values)/length/1000}


def memory():
    result = {'python_heap_used_bytes': gc.mem_alloc()}
    for name in ('sys_heap', 'sys_page', 'sys_mmz'):
        if hasattr(gc, name):
            try:
                _, used, _ = getattr(gc, name)()
                result[name+'_used_bytes'] = used
            except (ValueError, OSError, TypeError):
                pass
    return result


def digest(path):
    import binascii
    import hashlib
    h = hashlib.sha256()
    buffer = bytearray(4096)
    view = memoryview(buffer)
    with open(path, 'rb') as stream:
        while True:
            size = stream.readinto(buffer)
            if not size:
                break
            h.update(view[:size])
    return binascii.hexlify(h.digest()).decode()


def run(root='/sdcard/topface_trial', frames=600, display=True):
    import os
    import sys
    if sys.implementation.name != 'micropython':
        raise RuntimeError('Run on CanMV K230')
    if type(frames) is not int or not 5 <= frames <= 18000:
        raise ValueError('Frame count must be 5 to 18,000')
    from media.sensor import Sensor, CAM_CHN_ID_0
    from media.media import MediaManager
    from media.display import Display
    import image
    import nncase_runtime as nn
    from ulab import numpy as np

    with open(root+'/identity.json') as stream:
        identity = json.load(stream)
    path = root+'/topface.kmodel'
    if digest(path) != identity['kmodel_sha256']:
        raise ValueError('Model digest differs from deployment identity')
    width, height = identity['model_wh']
    positions = identity['output_shape'][1]
    if (width, height) not in ((224, 128), (320, 192), (384, 224)):
        raise ValueError('Invalid model dimensions')
    if identity['output_shape'] != [1, positions, 50] or identity['input'] != 'NCHW_RGB_UINT8':
        raise ValueError('Unexpected model input or output contract')
    if hasattr(os, 'exitpoint'):
        os.exitpoint(os.EXITPOINT_ENABLE)
    scale = min(width/640, height/360)
    rw = round(640*scale)
    rh = round(360*scale)
    left = (width-rw)//2
    top = (height-rh)//2
    sensor = None
    initialized_media = False
    initialized_display = False
    kpu = builder = ai2d = destination = backing = overlay = None
    frame = raw = output = source = None
    last_observation = None
    stage_names = ('capture', 'ai2d', 'kpu', 'decode', 'display', 'ai2d_kpu', 'through_postprocess', 'total')
    samples = {name: [] for name in stage_names}
    dropped = 0
    memory_peak = memory()
    cleanup_errors = []
    status = 'STARTED'
    error_text = None
    start = time.ticks_ms()
    measurement_start = measurement_end = None
    measured_attempts = 0
    try:
        kpu = nn.kpu()
        kpu.load_kmodel(path)
        backing = np.zeros((1, 3, height, width), dtype=np.uint8)
        destination = nn.from_numpy(backing)
        kpu.set_input_tensor(0, destination)
        ai2d = nn.ai2d()
        ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        ai2d.set_pad_param(True, [0, 0, 0, 0, top, height-rh-top, left, width-rw-left], 0, [114]*3)
        ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        builder = ai2d.build([1, 3, 360, 640], [1, 3, height, width])
        if display:
            Display.init(Display.ST7701, width=800, height=480, fps=30, to_ide=False)
            initialized_display = True
            overlay = image.Image(640, 360, image.RGB565)
        sensor = Sensor(id=2, width=1920, height=1080, fps=30)
        sensor.reset()
        sensor.set_framesize(width=640, height=360, crop=False, chn=CAM_CHN_ID_0)
        sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_0)
        sensor.set_hmirror(False)
        sensor.set_vflip(False)
        MediaManager.init()
        initialized_media = True
        sensor.run()
        for step in range(frames+5):
            if step == 5:
                measurement_start = time.ticks_ms()
            if step >= 5:
                measured_attempts += 1
            if hasattr(os, 'exitpoint'):
                os.exitpoint()
            start_frame = time.ticks_us()
            try:
                frame = sensor.snapshot(chn=CAM_CHN_ID_0, timeout=1000 if step == 0 else 150, auto_reboot=False)
            except RuntimeError as error:
                if 'timeout' not in str(error).lower():
                    raise
                dropped += int(step >= 5)
                continue
            if frame is None:
                dropped += int(step >= 5)
                continue
            if frame.width()!=640 or frame.height()!=360:
                raise ValueError('CSI2 frame dimensions differ from the model contract')
            captured = time.ticks_us()
            source = nn.from_numpy(frame.to_numpy_ref())
            builder.run(source, destination)
            prepared = time.ticks_us()
            kpu.run()
            inferred = time.ticks_us()
            output = kpu.get_output_tensor(0)
            raw = output.to_numpy()
            if tuple(raw.shape) != (1, positions, 50):
                raise ValueError('KPU output shape differs from the contract')
            detections = decode(lambda point, channel: raw[0, point, channel],
                                width=width, height=height, positions=positions,
                                score_threshold=identity['score_threshold'], nms_iou=identity['nms_iou'])
            last_observation = detections
            decoded = time.ticks_us()
            raw = output = source = None
            if display:
                overlay.draw_image(frame.to_rgb565(), 0, 0)
                overlay.draw_string(8,8,detections['capacity_state'],color=(255,255,255))
                for target in detections['detections']:
                    x1, y1, x2, y2 = target['bbox_xyxy_visible']
                    overlay.draw_rectangle(int(x1), int(y1), int(x2-x1), int(y2-y1), color=(0,255,0), thickness=2)
                    cx, cy = target['center_xy_projected_3d']
                    if 0 <= cx < 640 and 0 <= cy < 360:
                        overlay.draw_cross(int(cx), int(cy), color=(255,0,255))
                    ellipse=target['ellipse']
                    if ellipse is not None:
                        ex,ey=ellipse['conic_center_xy']
                        theta=ellipse['orientation_deg']*math.pi/180
                        previous=None
                        for point in range(25):
                            angle=point*2*math.pi/24
                            dx=ellipse['semi_major']*math.cos(angle)
                            dy=ellipse['semi_minor']*math.sin(angle)
                            pixel=(int(ex+dx*math.cos(theta)-dy*math.sin(theta)),
                                   int(ey+dx*math.sin(theta)+dy*math.cos(theta)))
                            if previous is not None:
                                overlay.draw_line(previous[0],previous[1],pixel[0],pixel[1],color=(255,255,0))
                            previous=pixel
                Display.show_image(overlay, x=80, y=60)
            shown = time.ticks_us()
            if step >= 5:
                boundaries = ((captured, start_frame), (prepared, captured), (inferred, prepared),
                              (decoded, inferred), (shown, decoded), (inferred, captured),
                              (decoded, start_frame), (shown, start_frame))
                for name, (end, begin) in zip(stage_names, boundaries):
                    samples[name].append(time.ticks_diff(end, begin))
            if step % 30 == 0:
                for key, used in memory().items():
                    memory_peak[key] = max(used, memory_peak.get(key, 0))
            frame = None
        status = ('RUN_COMPLETE' if not dropped else 'RUN_COMPLETE_WITH_CAPTURE_LOSSES') if samples['total'] else 'NO_SUCCESSFUL_MEASUREMENTS'
    except Exception as error:
        status = 'RUN_FAILED'
        error_text = str(error)
    finally:
        if measurement_start is not None:
            measurement_end = time.ticks_ms()
        raw = output = source = frame = None
        for label, cleanup in (('sensor', sensor.stop if sensor is not None else None),
                               ('display', Display.deinit if initialized_display else None),
                               ('media', MediaManager.deinit if initialized_media else None)):
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    cleanup_errors.append(label+': '+str(error))
        kpu = builder = ai2d = destination = backing = overlay = None
        gc.collect()
        if hasattr(nn,'shrink_memory_pool'):
            try:
                nn.shrink_memory_pool()
            except Exception as error:
                cleanup_errors.append('pool: '+str(error))
    seconds = time.ticks_diff(time.ticks_ms(), start)/1000
    measurement_seconds = (time.ticks_diff(measurement_end, measurement_start)/1000
                           if measurement_start is not None else None)
    report = {'status': status, 'error': error_text, 'runtime': str(nn.version()), 'display': display,
              'model_sha256': identity['kmodel_sha256'], 'capture_timeouts': dropped,
              'displayed_frames': len(samples['total']) if display else 0,
              'requested_measurement_frames': frames, 'elapsed_seconds_including_setup': seconds,
              'measurement_attempts': measured_attempts, 'successful_measurement_frames':len(samples['total']),
              'measurement_wall_seconds':measurement_seconds,
              'effective_camera_to_result_fps':(len(samples['total'])/measurement_seconds
                  if measurement_seconds is not None and measurement_seconds > 0 else None),
              'timing': {name: summarize(values) for name, values in samples.items()},
              'peak_sampled_memory': memory_peak, 'memory_after_cleanup': memory(),
              'cleanup_errors': cleanup_errors, 'last_observation': last_observation}
    if samples['total']:
        report['successful_frame_active_time_fps'] = 1000000*len(samples['total'])/sum(samples['total'])
    print('TOPFACE_BOARD_RESULT '+json.dumps(report))
    return report
