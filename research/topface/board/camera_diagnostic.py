"""Isolated, bounded camera and display probes; no model or startup writes."""
import gc
import json
import os
import time


def record_file(path):
    import hashlib
    import binascii
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(4096)
            if not block:
                break
            h.update(block)
    return {'path': path, 'bytes': os.stat(path)[6],
            'sha256': binascii.hexlify(h.digest()).decode()}


def run(kind, root='/sdcard/cdiag1116', seconds=20):
    if kind not in ('pattern', 'csi_planar', 'csi_rgb565', 'uvc'):
        raise ValueError('Unknown diagnostic source')
    if not 5 <= seconds <= 120:
        raise ValueError('Diagnostic must be bounded to 5..120 seconds')
    import image
    from media.display import Display
    from media.media import MediaManager
    sensor = None
    started_uvc = False
    display_ready = media_ready = False
    report = {'kind': kind, 'frames': 0, 'capture_errors': [], 'files': [],
              'status': 'STARTED', 'cleanup_errors': []}
    try:
        Display.init(Display.ST7701, width=800, height=480, fps=30, to_ide=True)
        display_ready = True
        if kind.startswith('csi_'):
            from media.sensor import Sensor, CAM_CHN_ID_0
            sensor = Sensor(id=2, width=1920, height=1080, fps=30)
            sensor.reset()
            sensor.set_framesize(width=640, height=360, crop=False, chn=CAM_CHN_ID_0)
            sensor.set_pixformat(Sensor.RGBP888 if kind == 'csi_planar' else Sensor.RGB565,
                                chn=CAM_CHN_ID_0)
            sensor.set_hmirror(False)
            sensor.set_vflip(False)
            MediaManager.init()
            media_ready = True
            sensor.run()
            report['camera_id'] = 2
            for _ in range(20):
                sensor.snapshot(chn=CAM_CHN_ID_0, timeout=1000, auto_reboot=False)
        elif kind == 'uvc':
            from media.uvc import UVC
            connected, device = UVC.probe()
            report.update(connected=connected, device=device)
            if not connected:
                raise RuntimeError('NO_UVC_DEVICE')
            report['modes'] = [str(mode) for mode in UVC.list_video_mode()]
            ok, mode = UVC.select_video_mode(UVC.video_mode(640,480,UVC.FOURCC_MJPEG,30))
            report.update(mode_selected=ok, actual_mode=str(mode))
            if not ok:
                raise RuntimeError('UVC_MODE_NOT_SELECTED')
            started_uvc = True
            report['start_result'] = UVC.start(cvt=False)
        if kind == 'pattern':
            pattern = image.Image(800, 480, image.RGB565)
            colors = [(255,255,255),(255,255,0),(0,255,255),(0,255,0),
                      (255,0,255),(255,0,0),(0,0,255),(0,0,0)]
            for i, color in enumerate(colors):
                pattern.draw_rectangle(i*100,0,100,480,color=color,fill=True)
            pattern.draw_string_advanced(12,12,28,'DISPLAY TEST',color=(0,0,0))
            name = root+'/pattern.bmp'
            pattern.save(name)
            report['files'].append(record_file(name))
        deadline = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), deadline) < seconds*1000:
            os.exitpoint()
            source = converted = None
            if kind == 'pattern':
                converted = pattern
            else:
                try:
                    source = (UVC.snapshot() if kind == 'uvc' else
                              sensor.snapshot(chn=CAM_CHN_ID_0,timeout=1000,auto_reboot=False))
                    if source is None:
                        report['capture_errors'].append('NO_FRAME')
                        continue
                    if report['frames'] == 0:
                        report['source'] = {'width': source.width(), 'height': source.height(),
                                            'format': source.format(), 'bytes': source.size()}
                        ext = '.jpg' if kind == 'uvc' else '.raw'
                        name = root+'/'+kind+'-source'+ext
                        with open(name,'wb') as stream:
                            stream.write(source.bytearray())
                        report['files'].append(record_file(name))
                    converted = source if kind == 'csi_rgb565' else source.to_rgb565()
                    if report['frames'] == 0:
                        name = root+'/'+kind+'-converted.bmp'
                        converted.save(name)
                        report['files'].append(record_file(name))
                except Exception as error:
                    report['capture_errors'].append(str(error))
                    if len(report['capture_errors']) >= 5:
                        raise
                    continue
            Display.show_image(converted, x=0 if kind=='pattern' else 80,
                                y=0 if kind in ('pattern','uvc') else 60)
            report['frames'] += 1
            source = converted = None
            gc.collect()
            time.sleep_ms(50)
        report['status'] = 'COMPLETE'
    except BaseException as error:
        report.update(status='FAILED', error=str(error))
    finally:
        for label, cleanup in (
                ('sensor', sensor.stop if sensor is not None else None),
                ('uvc', UVC.stop if started_uvc else None),
                ('display', Display.deinit if display_ready else None),
                ('media', MediaManager.deinit if media_ready else None)):
            if cleanup is not None:
                try:
                    cleanup()
                except BaseException as error:
                    report['cleanup_errors'].append(label+': '+str(error))
        gc.collect()
        with open(root+'/'+kind+'-report.json','w') as stream:
            json.dump(report, stream)
        print('CAMERA_DIAGNOSTIC '+json.dumps(report))
    return report
