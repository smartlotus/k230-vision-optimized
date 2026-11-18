import sys
sys.path.insert(0,'/sdcard/tfp1116')
sys.path.insert(0,'/sdcard/rn1117a')
"""USB-to-model engineering smoke: no labels, accuracy, geometry or control outputs."""
import gc
import json
import os
import time


def run(run_id, seconds=60, with_topface=True, preview=True,
        road_root='/sdcard/rn1117a', top_root='/sdcard/tfp1116/model'):
    import image
    from ulab import numpy as np
    from media.uvc import UVC
    from media.media import MediaManager
    from media.display import Display
    from board_smoke import RoadNetTrial, Latency, sample_memory
    from k230.topface_adapter import TopFaceBackend
    from k230.topface_service import resources

    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise ValueError('Use a bounded 1..300 second engineering smoke')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('Explicit run identity required')
    result = {'run_id': run_id, 'scope': 'USB camera, model ABI, full-output structural decode, display and cleanup only',
              'accuracy_test': 'NOT_RUN_BY_USER_SCOPE', 'road_quality_test': 'NOT_RUN_BY_USER_SCOPE',
              'csi_qr_control': 'NOT_IN_SCOPE', 'requested_seconds': seconds,
              'with_topface': with_topface, 'preview': preview, 'attempts': 0, 'frames': 0,
              'roadnet_inferences': 0, 'postprocess_frames': 0, 'topface_inferences': 0,
              'shared_inferences': 0, 'displayed_frames': 0, 'capture_failures': 0,
              'capture_errors': [], 'cleanup_errors': [], 'memory_sampling_failures': 0,
              'active_memory_samples': 0, 'active_pool_peaks': {},
              'model_input': [1, 3, 192, 320], 'model_output': [1, 7, 192, 320],
              'firmware': str(os.uname()), 'status': 'STARTED'}
    stages = ('capture', 'usb_rgb', 'roadnet', 'roadnet_ai2d', 'roadnet_kpu',
              'roadnet_readback', 'roadnet_postprocess', 'topface', 'display', 'gc', 'cycle')
    latencies = {name: Latency() for name in stages}
    road = top = raw = packed = array = planar = owned = shown = None
    src = tensor = logits = labels = content = None
    uvc_attempted = display_attempted = media_attempted = False
    started = None
    sample_memory(result, resources, 'initial')
    try:
        connected, device = UVC.probe()
        result['usb_device'] = str(device)
        if not connected:
            raise RuntimeError('UVC not enumerated')
        result['supported_modes'] = str(UVC.list_video_mode()) if hasattr(UVC, 'list_video_mode') else 'UNAVAILABLE'
        success, mode = UVC.select_video_mode(UVC.video_mode(1280, 720, UVC.FOURCC_MJPEG, 30))
        if not success or (mode.width, mode.height, mode.fourcc) != (1280, 720, UVC.FOURCC_MJPEG):
            raise ValueError('Expected frozen USB 1280x720 MJPEG input')
        result['negotiated'] = {'width': mode.width, 'height': mode.height, 'fourcc': mode.fourcc, 'fps': mode.fps}
        road = RoadNetTrial(road_root)
        result['roadnet_sha256'] = road.identity['kmodel_sha256']
        if with_topface:
            top = TopFaceBackend(top_root)
            result['topface_sha256'] = top.identity['kmodel_sha256']
        if preview:
            display_attempted = True
            Display.init(Display.ST7701, width=800, height=480, to_ide=True)
        media_attempted = True
        MediaManager.init()
        uvc_attempted = True
        if UVC.start(cvt=False) is False:
            raise RuntimeError('UVC start failed')
        print('USB_ENGINEERING_START', run_id)
        started = last_memory = time.ticks_ms()
        consecutive = 0
        sample_memory(result, resources, 'active')
        while time.ticks_diff(time.ticks_ms(), started) < seconds * 1000:
            os.exitpoint()
            cycle = time.ticks_us()
            result['attempts'] += 1
            before = time.ticks_us()
            try:
                raw = UVC.snapshot(300)
                if raw is None:
                    raise RuntimeError('UVC snapshot returned no frame')
            except RuntimeError as error:
                result['capture_failures'] += 1
                consecutive += 1
                if len(result['capture_errors']) < 8:
                    result['capture_errors'].append({'attempt': result['attempts'], 'message': str(error)})
                if consecutive >= 3:
                    raise
                gc.collect()
                time.sleep_ms(100)
                continue
            latencies['capture'].add(time.ticks_diff(time.ticks_us(), before))
            consecutive = 0
            if (raw.width(), raw.height()) != (1280, 720):
                raise ValueError('Acquired frame differs from negotiated size')
            result['frames'] += 1
            before = time.ticks_us()
            packed = raw.to_rgb888(x_size=640, y_size=360)
            array = packed.to_numpy_ref()
            if tuple(array.shape) != (360, 640, 3):
                raise ValueError('Unexpected packed RGB array shape')
            planar = array.reshape((360 * 640, 3)).transpose().copy().reshape((3, 360, 640))
            owned = image.Image(640, 360, image.RGBP888, alloc=image.ALLOC_REF, data=planar)
            latencies['usb_rgb'].add(time.ticks_diff(time.ticks_us(), before))
            before = time.ticks_us()
            src = road.nn.from_numpy(owned.to_numpy_ref())
            road.builder.run(src, road.destination)
            prepared = time.ticks_us()
            road.kpu.run()
            inferred = time.ticks_us()
            tensor = road.kpu.get_output_tensor(0)
            logits = tensor.to_numpy()
            if tuple(logits.shape) != (1, 7, 192, 320):
                raise ValueError('Unexpected RoadNet output tensor shape')
            if int(np.sum(np.isfinite(logits))) != 430080:
                raise ValueError('Nonfinite RoadNet output')
            read = time.ticks_us()
            # Structural decoder exercise only. No class proportions, ground-truth
            # comparisons, road contours, safety geometry or actuator outputs.
            labels = np.argmax(logits.reshape((7, 192, 320)), axis=0)
            content = labels[6:186, :]
            if tuple(content.shape) != (180, 320) or int(np.min(content)) < 0 or int(np.max(content)) > 6:
                raise ValueError('Unexpected decoded content mask contract')
            processed = time.ticks_us()
            result['roadnet_inferences'] += 1
            result['postprocess_frames'] += 1
            result['decoded_content_shape'] = list(content.shape)
            latencies['roadnet_ai2d'].add(time.ticks_diff(prepared, before))
            latencies['roadnet_kpu'].add(time.ticks_diff(inferred, prepared))
            latencies['roadnet_readback'].add(time.ticks_diff(read, inferred))
            latencies['roadnet_postprocess'].add(time.ticks_diff(processed, read))
            latencies['roadnet'].add(time.ticks_diff(processed, before))
            content = labels = logits = tensor = src = None
            if top is not None:
                before = time.ticks_us()
                decoded = top.infer(owned)
                if not isinstance(decoded.get('detections'), list):
                    raise ValueError('Unexpected TopFace decoder contract')
                result['topface_inferences'] += 1
                result['shared_inferences'] += 1
                decoded = None
                latencies['topface'].add(time.ticks_diff(time.ticks_us(), before))
            if preview:
                before = time.ticks_us()
                # Display packed RGB, preserving the independently owned model frame.
                shown = packed.to_rgb565()
                shown.draw_string_advanced(8, 8, 20, 'ENGINEERING ONLY  USB -> KPU -> OUTPUT OK', color=(255, 255, 255))
                shown.draw_string_advanced(8, 34, 20, 'frames ' + str(result['frames']), color=(255, 255, 255))
                Display.show_image(shown, x=80, y=60)
                result['displayed_frames'] += 1
                latencies['display'].add(time.ticks_diff(time.ticks_us(), before))
            shown = owned = planar = array = packed = raw = None
            before = time.ticks_us()
            gc.collect()
            latencies['gc'].add(time.ticks_diff(time.ticks_us(), before))
            now = time.ticks_ms()
            if time.ticks_diff(now, last_memory) >= 1000:
                sample_memory(result, resources, 'active')
                last_memory = now
            latencies['cycle'].add(time.ticks_diff(time.ticks_us(), cycle))
        if not result['frames'] or result['roadnet_inferences'] != result['frames']:
            raise ValueError('Incomplete camera-to-output frame count')
        if with_topface and result['shared_inferences'] != result['frames']:
            raise ValueError('Incomplete shared-frame model count')
        if preview and result['displayed_frames'] != result['frames']:
            raise ValueError('Incomplete display count')
        result['status'] = 'ENGINEERING_MEASURED' if not result['capture_failures'] else 'ENGINEERING_WITH_CAPTURE_LOSSES'
    except BaseException as error:
        result['status'] = 'ERROR'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        result['elapsed_seconds'] = time.ticks_diff(time.ticks_ms(), started) / 1000 if started is not None else 0
        content = labels = logits = tensor = src = shown = owned = planar = array = packed = raw = None
        gc.collect()
        operations = []
        if uvc_attempted:
            operations.append(('uvc', UVC.stop))
        if display_attempted:
            operations.append(('display', Display.deinit))
        if media_attempted:
            operations.append(('media', MediaManager.deinit))
        if road is not None:
            operations.append(('roadnet', road.close))
        if top is not None:
            operations.append(('topface', top.close))
        for name, operation in operations:
            try:
                operation()
            except BaseException as error:
                result['cleanup_errors'].append({'stage': name, 'message': str(error)})
        if result['cleanup_errors']:
            result['status'] = 'CLEANUP_UNCONFIRMED'
        sample_memory(result, resources, 'final')
        result['fps'] = result['frames'] / result['elapsed_seconds'] if result['elapsed_seconds'] else None
        result['latency'] = {name: hist.report() for name, hist in latencies.items()}
        print('USB_ENGINEERING_RESULT:' + json.dumps(result))
    return result

run('new-sd-chain120-20261118',120,with_topface=True,preview=True,road_root='/sdcard/rn1117a',top_root='/sdcard/tfp1116/model')
