"""Isolated OV2735 → AI2D → CameraXY KModel → CPU bench for A and C."""
import json

from board_temporal import BenchDecoder


class RGBWindow:
    """Past-to-current planar history. Timeouts never insert an image."""
    def __init__(self, packed, length):
        if length not in (1, 4, 8) or tuple(packed.shape) != (3 * length, 96, 160):
            raise ValueError('Invalid CameraXY history shape')
        self.packed = packed
        self.length = length
        self.filled = False

    def clear(self):
        self.filled = False

    def push(self, frame):
        if tuple(frame.shape) != (3, 96, 160):
            raise ValueError('Invalid AI2D frame shape')
        if self.filled:
            for i in range(self.length - 1):
                self.packed[i * 3:(i + 1) * 3] = self.packed[(i + 1) * 3:(i + 2) * 3]
            self.packed[(self.length - 1) * 3:self.length * 3] = frame
        else:
            for i in range(self.length):
                self.packed[i * 3:(i + 1) * 3] = frame
            self.filled = True


def run(model_root, run_id, expected_usb_device, frames=120, mode='filtered'):
    """Explicit CanMV entry point, independent of vehicle control and boot."""
    import gc
    import os
    import sys
    import time

    if sys.implementation.name != 'micropython':
        raise RuntimeError('Run on CanMV MicroPython with an idle OV2735')
    if not (model_root.startswith('/sdcard/k230_temporal_v1_')
            or model_root.startswith('/data/k230_temporal_v1_')
            or model_root.startswith('/tmp/k230_temporal_v1_')) or any(
        part in ('', '.', '..') for part in model_root.split('/')[1:]
    ):
        raise ValueError('Use a dedicated k230_temporal_v1_ test directory')
    if (not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in run_id)
            or not expected_usb_device or type(frames) is not int or not 1 <= frames <= 3000):
        raise ValueError('Specify a unique report ID, the verified USB ID and 1..3000 attempts')
    report_path = model_root + '/' + run_id + '.json'
    for path in (report_path, report_path + '.partial'):
        try:
            os.stat(path)
        except OSError:
            continue
        raise ValueError('Benchmark report or partial already exists')

    from media.media import MediaManager
    from media.uvc import UVC
    import nncase_runtime as nn
    from ulab import numpy as np
    from board_helpers import Latency, digest, resources, sample_memory

    with open(model_root + '/deployment.json') as stream:
        deployment = json.load(stream)
    identity = deployment['identity']
    config = deployment['temporal_filter']
    model_path = model_root + '/camera_xy.kmodel'
    expected = deployment['kmodel']['kmodel_sha256']
    length = identity['window']
    shape = (1, 3 * length, 96, 160)
    if (config.get('acceptance') is not None or list(shape) != identity['input_rgb_uint8_nchw']
            or digest(model_path) != expected):
        raise ValueError('Dedicated model identity or qualification differs')
    centers = identity['map_centers_mm']
    times = {name: Latency() for name in ('capture', 'rgb', 'ai2d', 'history', 'kpu', 'decode', 'filter', 'cycle')}
    decoder = BenchDecoder(centers, config, mode, time.ticks_us, time.ticks_diff, times['filter'])
    result = dict(status='RUNNING', mode=mode, device=str(os.uname()), nncase_runtime=nn.version(),
                  expected_nncase_runtime=deployment['kmodel']['compiler']['nncase'],
                  requested_frames=frames, frames=0,
                  dropped=0, unlocalized=0, history_window=length, history_resets=0,
                  kmodel_sha256=expected, model_bytes=os.stat(model_path)[6], cleanup_errors=[],
                  active_pool_peaks={}, active_memory_samples=0, memory_sampling_failures=0,
                  native_ai2d_pixel_equivalence='NOT RUN', real_camera_accuracy='NOT RUN')
    kpu = ai2d = builder = backing = model_input = frame_input = frame_buffer = history = None
    src = raw_image = rgb_image = pixels = planar = output_tensor = None
    media_started = usb_started = False
    started = None

    try:
        sample_memory(result, resources, 'initial')
        found, usb_id = UVC.probe()
        if not found or usb_id != expected_usb_device:
            raise ValueError('Selected UVC source differs from the verified OV2735')
        ok, negotiated = UVC.select_video_mode(UVC.video_mode(1280, 720, UVC.FOURCC_MJPEG, 30))
        if not ok or (negotiated.width, negotiated.height, negotiated.fourcc) != (1280, 720, UVC.FOURCC_MJPEG):
            raise ValueError('OV2735 mode differs from the benchmark capture contract')
        result['negotiated_uvc'] = dict(device=usb_id, size=[negotiated.width, negotiated.height],
                                        fourcc=negotiated.fourcc, fps=negotiated.fps)
        kpu = nn.kpu()
        kpu.load_kmodel(model_path)
        if kpu.inputs_size() != 1 or kpu.outputs_size() != 1:
            raise ValueError('Unexpected KPU graph arity')
        if length == 1:
            backing = np.zeros(shape, dtype=np.uint8)
            model_input = nn.from_numpy(backing)
            kpu.set_input_tensor(0, model_input)
            frame_input = model_input
        else:
            frame_buffer = np.zeros((1, 3, 96, 160), dtype=np.uint8)
            frame_input = nn.from_numpy(frame_buffer)
            history = RGBWindow(np.zeros(shape[1:], dtype=np.uint8), length)
        ai2d = nn.ai2d()
        ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        ai2d.set_pad_param(True, [0, 0, 0, 0, 3, 3, 0, 0], 0, [114] * 3)
        ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        builder = ai2d.build([1, 3, 360, 640], [1, 3, 96, 160])
        media_started = True
        MediaManager.init()
        usb_started = True
        if UVC.start(cvt=False) is False:
            raise RuntimeError('OV2735 failed to start')
        started = time.ticks_ms()
        consecutive_drops = 0
        for index in range(frames):
            os.exitpoint()
            begin = time.ticks_us()
            raw_image = UVC.snapshot(300)
            captured = time.ticks_us()
            times['capture'].add(time.ticks_diff(captured, begin))
            if raw_image is None:
                result['dropped'] += 1
                consecutive_drops += 1
                lost = decoder(None, centers, None, None, captured_tick=captured)
                result['last_prediction'] = lost
                if history is not None and consecutive_drops == 5:
                    history.clear()
                    result['history_resets'] += 1
                times['cycle'].add(time.ticks_diff(time.ticks_us(), begin))
                if result['frames'] == 0 and consecutive_drops >= 10:
                    raise RuntimeError('No valid OV2735 capture')
                continue

            consecutive_drops = 0
            rgb_image = raw_image.to_rgb888(x_size=640, y_size=360)
            pixels = rgb_image.to_numpy_ref()
            if tuple(pixels.shape) != (360, 640, 3):
                raise ValueError('OV2735 RGB conversion differs from the model contract')
            planar = pixels.reshape((360 * 640, 3)).transpose().copy().reshape((1, 3, 360, 640))
            before_ai2d = time.ticks_us()
            times['rgb'].add(time.ticks_diff(before_ai2d, captured))
            src = nn.from_numpy(planar)
            builder.run(src, frame_input)
            after_ai2d = time.ticks_us()
            times['ai2d'].add(time.ticks_diff(after_ai2d, before_ai2d))
            if history is not None:
                history.push(frame_input.to_numpy()[0])
                model_input = nn.from_numpy(history.packed.reshape(shape))
                kpu.set_input_tensor(0, model_input)
            before_kpu = time.ticks_us()
            times['history'].add(time.ticks_diff(before_kpu, after_ai2d))
            kpu.run()
            before_decode = time.ticks_us()
            times['kpu'].add(time.ticks_diff(before_decode, before_kpu))
            output_tensor = kpu.get_output_tensor(0)
            values = [float(value) for value in output_tensor.to_numpy()[0]]
            answer = decoder(values, centers, None, None, captured_tick=captured)
            result['last_prediction'] = answer
            times['decode'].add(time.ticks_diff(time.ticks_us(), before_decode))
            result['frames'] += 1
            result['unlocalized'] += int(not answer['position_valid'])
            src = raw_image = rgb_image = pixels = planar = output_tensor = None
            if index % 10 == 0:
                sample_memory(result, resources, 'active')
            times['cycle'].add(time.ticks_diff(time.ticks_us(), begin))
        result['status'] = 'MEASURED' if result['frames'] else 'NO_FRAMES'
    except BaseException as error:
        result.update(status='RUN_FAILED', error=str(error))
        raise
    finally:
        if started is not None:
            elapsed = time.ticks_diff(time.ticks_ms(), started) / 1000
            result['wall_seconds'] = elapsed
            result['camera_to_result_fps'] = result['frames'] / elapsed if elapsed > 0 else None
        for name, enabled, close in (('uvc', usb_started, UVC.stop), ('media', media_started, MediaManager.deinit)):
            if enabled:
                try:
                    close()
                except Exception as error:
                    result['cleanup_errors'].append([name, str(error)])
        if result['cleanup_errors']:
            result['status'] = 'CLEANUP_UNCONFIRMED'
        src = raw_image = rgb_image = pixels = planar = output_tensor = None
        kpu = ai2d = builder = backing = model_input = frame_input = frame_buffer = history = None
        gc.collect()
        try:
            nn.shrink_memory_pool()
        except Exception as error:
            result['cleanup_errors'].append(['nncase_memory_pool', str(error)])
            result['status'] = 'CLEANUP_UNCONFIRMED'
        sample_memory(result, resources, 'final')
        result['latency'] = {name: histogram.report() for name, histogram in times.items()}
        result['temporal_states'] = decoder.states
        with open(report_path + '.partial', 'w') as stream:
            stream.write(json.dumps(result))
        os.rename(report_path + '.partial', report_path)
    return result
