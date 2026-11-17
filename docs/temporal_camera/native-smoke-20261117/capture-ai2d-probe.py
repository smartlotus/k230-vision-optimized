"""RAM-only USB capture and preprocessing probe; no model, UART, or SD writes."""
import gc
import json
import os
import sys
import time
from media.media import MediaManager
from media.uvc import UVC
import nncase_runtime as nn
from ulab import numpy as np

r = dict(nonce='TEMPORAL_CAPTURE_AI2D_B_20261117', status='STARTED', attempts=10,
         frames=0, dropped=0, cleanup_errors=[], kpu_inferences=0, filter_updates=0,
         accuracy='NOT_RUN', road_test='NOT_RUN', board_files_written=0,
         initial_heap_free=gc.mem_free())
media_started = usb_started = False
raw = rgb = pixels = planar = src = output = tensor = ai2d = builder = None
try:
    found, device = UVC.probe()
    r['usb'] = device
    if not found or device != 'openaicam#openaicam':
        raise ValueError('Selected UVC source differs')
    ok, mode = UVC.select_video_mode(UVC.video_mode(1280, 720, UVC.FOURCC_MJPEG, 30))
    if not ok or (mode.width, mode.height, mode.fourcc) != (1280, 720, UVC.FOURCC_MJPEG):
        raise ValueError('UVC mode differs')
    r['mode'] = dict(width=mode.width, height=mode.height, fourcc=mode.fourcc, fps=mode.fps)
    output = np.zeros((1, 3, 96, 160), dtype=np.uint8)
    tensor = nn.from_numpy(output)
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, 3, 3, 0, 0], 0, [114] * 3)
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    builder = ai2d.build([1, 3, 360, 640], [1, 3, 96, 160])
    media_started = True
    MediaManager.init()
    usb_started = True
    if UVC.start(cvt=False) is False:
        raise RuntimeError('UVC start failed')
    begin = time.ticks_ms()
    for index in range(r['attempts']):
        os.exitpoint()
        raw = UVC.snapshot(300)
        if raw is None:
            r['dropped'] += 1
            continue
        rgb = raw.to_rgb888(x_size=640, y_size=360)
        pixels = rgb.to_numpy_ref()
        r['rgb_shape'] = list(pixels.shape)
        if tuple(pixels.shape) != (360, 640, 3):
            raise ValueError('Unexpected native RGB layout')
        planar = pixels.reshape((360 * 640, 3)).transpose().copy().reshape((1, 3, 360, 640))
        src = nn.from_numpy(planar)
        builder.run(src, tensor)
        r['input_shape'] = list(output.shape)
        r['input_min'] = int(np.min(output))
        r['input_max'] = int(np.max(output))
        r['frames'] += 1
        raw = rgb = pixels = planar = src = None
        gc.collect()
    r['seconds'] = time.ticks_diff(time.ticks_ms(), begin) / 1000
    r['status'] = 'CAPTURE_AND_AI2D_PASS' if r['frames'] == r['attempts'] else 'CAPTURE_INCOMPLETE'
except BaseException as error:
    r['status'] = 'FAILED'
    r['error'] = str(error)
    sys.print_exception(error)
finally:
    raw = rgb = pixels = planar = src = None
    gc.collect()
    for name, active, close in (('uvc', usb_started, UVC.stop), ('media', media_started, MediaManager.deinit)):
        if active:
            try:
                close()
            except Exception as error:
                r['cleanup_errors'].append([name, str(error)])
    output = tensor = ai2d = builder = None
    gc.collect()
    r['final_heap_free'] = gc.mem_free()
    print('TEMPORAL_CAPTURE_REPORT:' + json.dumps(r))
