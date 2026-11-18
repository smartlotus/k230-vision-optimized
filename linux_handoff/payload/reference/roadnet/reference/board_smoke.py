"""Isolated CanMV RoadNet, TopFaceNet and QR acceptance measurements."""
import json
import math


class Latency:
    """Bounded 1 ms histogram with explicit overflow for long board runs."""
    def __init__(self):
        self.counts = [0] * 1002
        self.count = 0
        self.total_us = 0

    def add(self, elapsed_us):
        index = min(1001, max(0, (elapsed_us + 999) // 1000))
        self.counts[index] += 1
        self.count += 1
        self.total_us += elapsed_us

    def report(self):
        def percentile(p):
            remaining = (self.count * p + 99) // 100
            for elapsed, count in enumerate(self.counts):
                remaining -= count
                if remaining <= 0:
                    return elapsed if elapsed <= 1000 else None
        return {
            'samples': self.count,
            'mean_ms': self.total_us / self.count / 1000 if self.count else None,
            'p50_ms_upper': percentile(50) if self.count else None,
            'p95_ms_upper': percentile(95) if self.count else None,
            'p99_ms_upper': percentile(99) if self.count else None,
            'overflow_over_1000ms': self.counts[1001],
        }


class RoadNetTrial:
    """The same 640x360 owned planar RGB frame is reusable by TopFaceBackend."""
    def __init__(self, model_root):
        import nncase_runtime as nn
        from ulab import numpy as np
        from k230.topface_adapter import digest

        with open(model_root + '/identity.json') as stream:
            self.identity = json.load(stream)
        path = model_root + '/roadnet.kmodel'
        if digest(path) != self.identity['kmodel_sha256']:
            raise ValueError('RoadNet model SHA mismatch')
        if self.identity['input_rgb_uint8_nchw'] != [1, 3, 192, 320]:
            raise ValueError('RoadNet trial expects frozen uint8 RGB NCHW')
        self.nn = nn
        self.kpu = self.ai2d = self.builder = self.destination = self.backing = None
        self.timing_us = {}
        try:
            self.kpu = nn.kpu()
            self.kpu.load_kmodel(path)
            if self.kpu.inputs_size() != 1 or self.kpu.outputs_size() != 1:
                raise ValueError('RoadNet model requires one input and output')
            # Keep an owned input buffer, following the board-proven TopFace binding.
            self.backing = np.zeros((1, 3, 192, 320), dtype=np.uint8)
            self.destination = nn.from_numpy(self.backing)
            self.kpu.set_input_tensor(0, self.destination)
            self.ai2d = nn.ai2d()
            self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
            self.ai2d.set_pad_param(True, [0, 0, 0, 0, 6, 6, 0, 0], 0, [114] * 3)
            self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.builder = self.ai2d.build([1, 3, 360, 640], [1, 3, 192, 320])
        except BaseException:
            self.close()
            raise

    def infer(self, image):
        import time
        started = time.ticks_us()
        src = self.nn.from_numpy(image.to_numpy_ref())
        self.builder.run(src, self.destination)
        prepared = time.ticks_us()
        self.kpu.run()
        inferred = time.ticks_us()
        tensor = self.kpu.get_output_tensor(0)
        raw = tensor.to_numpy()
        if tuple(raw.shape) != tuple(self.identity['output_logits_nchw']):
            raise ValueError('Unexpected RoadNet output shape')
        # Sample the raw output contract, leaving geometric evaluation to calibrated hosts.
        center_y, center_x = raw.shape[2] // 2, raw.shape[3] // 2
        samples = [float(raw[0, i, center_y, center_x]) for i in range(7)]
        if not all(math.isfinite(value) for value in samples):
            raise ValueError('Nonfinite RoadNet sample logits')
        self.timing_us = {
            'ai2d': time.ticks_diff(prepared, started),
            'kpu': time.ticks_diff(inferred, prepared),
            'readback': time.ticks_diff(time.ticks_us(), inferred),
        }
        return samples

    def close(self):
        import gc
        self.kpu = self.ai2d = self.builder = self.destination = self.backing = None
        gc.collect()
        if hasattr(self.nn, 'shrink_memory_pool'):
            self.nn.shrink_memory_pool()


def frame_rate(frames, seconds):
    return frames / seconds if seconds > 0 else None


def sample_memory(result, resources, phase):
    try:
        memory = resources()
        if phase != 'active':
            result[phase + '_memory'] = memory
            return
        heap_free = memory['python_heap_free']
        result['minimum_python_heap_free'] = min(result.get('minimum_python_heap_free', heap_free), heap_free)
        for name, pool in memory.items():
            if isinstance(pool, dict) and pool['used'] > result['active_pool_peaks'].get(name, {}).get('used', -1):
                result['active_pool_peaks'][name] = pool
        result['active_memory_samples'] += 1
    except Exception as error:
        result['memory_sampling_failures'] += 1
        result['last_memory_sampling_failure'] = {'phase':phase, 'message':str(error)}


def run(mode, duration_seconds, report_path, roadnet_root=None, topface_root=None,
        startup_order='csi-first', expected_qr=None, csi_id=1):
    """Run on an idle board and return the report, optionally saving it to a file."""
    import gc
    import os
    import time
    import image
    from media.media import MediaManager
    from k230.topface_service import resources

    modes = ('uvc', 'csi', 'dual', 'roadnet', 'topface', 'dual-roadnet', 'both', 'qr')
    if mode not in modes or startup_order not in ('csi-first','uvc-first'):
        raise ValueError('Invalid measurement mode')
    if type(duration_seconds) is not int or not 1 <= duration_seconds <= 3600:
        raise ValueError('Duration must be 1 to 3600 seconds')
    if type(csi_id) is not int or csi_id not in (1, 2):
        raise ValueError('CSI ID must be 1 or 2')
    if report_path is not None and not report_path.startswith('/data/'):
        raise ValueError('Board reports must be stored under /data; keep models under /sdcard')
    if mode in ('roadnet','dual-roadnet','both') and not roadnet_root:
        raise ValueError('Provide frozen RoadNet model directory')
    if mode in ('topface','both') and not topface_root:
        raise ValueError('Provide frozen TopFace model directory')
    if report_path is not None:
        try:
            os.stat(report_path)
        except OSError:
            pass
        else:
            raise ValueError('Existing board report must be preserved')

    wants_usb = mode != 'csi'
    wants_csi = mode in ('csi','dual-roadnet','dual','both','qr')
    uvc = sensor = road = top = None
    media_attempted = uvc_attempted = sensor_configured = False
    result = {'mode':mode, 'requested_seconds':duration_seconds, 'startup_order':startup_order,
              'uvc_frames':0, 'csi_frames':0, 'uvc_drops':0, 'csi_drops':0,
              'cycles':0, 'shared_inferences':0, 'qr_frames':0,
              'qr_match':0, 'qr_wrong':0, 'qr_invalid':0, 'qr_ambiguous':0, 'qr_no_detection':0,
              'qr_expected_payload':expected_qr,
              'qr_valid_unscored':0, 'qr_fixture_bound':expected_qr is not None,
              'run_scope':'Isolated capture/inference timing, no vehicle-control outputs',
              'capture_loss_scope':'Acquisition requests returning no frame; producer frame sequence unavailable',
              'active_memory_samples':0, 'active_pool_peaks':{},
              'memory_sampling_failures':0, 'cleanup_errors':[]}
    if wants_csi:
        result['requested_csi_id'] = csi_id
    if hasattr(os, 'uname'):
        result['firmware_uname'] = str(os.uname())
    stages = ['uvc_capture', 'csi_capture', 'cycle']
    if mode in ('roadnet', 'topface', 'dual-roadnet', 'both'):
        stages.append('usb_rgb')
    if mode in ('roadnet', 'dual-roadnet', 'both'):
        stages.extend(('roadnet', 'roadnet_ai2d', 'roadnet_kpu', 'roadnet_readback'))
    if mode in ('topface', 'both'):
        stages.extend(('topface', 'topface_ai2d', 'topface_kpu', 'topface_decode'))
    if mode in ('qr', 'both'):
        stages.extend(('qr_grayscale', 'qr_decode'))
    latencies = {key:Latency() for key in stages}
    started = None
    try:
        sample_memory(result, resources, 'initial')
        if wants_csi:
            from media.sensor import Sensor
            sensor = Sensor(id=csi_id, width=1920, height=1080, fps=30)
            sensor_configured = True
            sensor.reset()
            sensor.set_framesize(width=640, height=480, crop=False)
            sensor.set_pixformat(Sensor.RGB565)
            sensor.set_hmirror(False)
            sensor.set_vflip(False)
            result['csi_configuration'] = {'id':csi_id,'width':640,'height':480,'format':'RGB565'}
        if wants_usb:
            from media.uvc import UVC
            uvc = UVC
            connected, device = uvc.probe()
            if not connected:
                raise RuntimeError('UVC not enumerated')
            success, negotiated = uvc.select_video_mode(uvc.video_mode(1280,720,uvc.FOURCC_MJPEG,30))
            if (not success or (negotiated.width,negotiated.height)!=(1280,720)
                    or negotiated.fourcc!=uvc.FOURCC_MJPEG):
                raise ValueError('OV2735 test requires 1280x720 MJPEG')
            result['uvc_device'] = device
            result['uvc_mode'] = str(negotiated)
            result['uvc_negotiated'] = {'width':negotiated.width, 'height':negotiated.height,
                                        'fourcc':negotiated.fourcc, 'fps':negotiated.fps}
        if mode in ('roadnet','dual-roadnet','both'):
            road = RoadNetTrial(roadnet_root)
            result['roadnet_sha256'] = road.identity['kmodel_sha256']
        if mode in ('topface','both'):
            from k230.topface_adapter import TopFaceBackend
            top = TopFaceBackend(topface_root)
            result['topface_sha256'] = top.identity['kmodel_sha256']
        if mode in ('qr','both'):
            from core.task_parser import parse_task
            if expected_qr is not None:
                parse_task(expected_qr)

        media_attempted = True
        MediaManager.init()
        for name in (('csi','uvc') if startup_order=='csi-first' else ('uvc','csi')):
            if name=='csi' and sensor is not None:
                sensor.run()
            if name=='uvc' and uvc is not None:
                uvc_attempted = True
                if uvc.start(cvt=False) is False:
                    raise RuntimeError('UVC start failed')

        print('ROADNET_SMOKE_START', mode, startup_order, report_path)
        started = time.ticks_ms()
        last_memory_sample = started
        sample_memory(result, resources, 'active')
        consecutive_uvc = consecutive_csi = 0
        while time.ticks_diff(time.ticks_ms(),started) < duration_seconds * 1000:
            os.exitpoint()
            cycle = time.ticks_us()
            if uvc is not None:
                t = time.ticks_us()
                raw = uvc.snapshot(300)
                latencies['uvc_capture'].add(time.ticks_diff(time.ticks_us(),t))
                if raw is None:
                    result['uvc_drops'] += 1
                    consecutive_uvc += 1
                else:
                    consecutive_uvc = 0
                    actual_size = (raw.width(), raw.height())
                    result['uvc_frame_wh'] = list(actual_size)
                    if actual_size != (1280,720):
                        raise ValueError('UVC snapshot size differs from negotiated mode')
                    result['uvc_frames'] += 1
                    if road or top:
                        t = time.ticks_us()
                        packed = raw.to_rgb888(x_size=640,y_size=360)
                        array = packed.to_numpy_ref()
                        if tuple(array.shape) != (360,640,3):
                            raise ValueError('UVC prepared RGB HWC shape mismatch')
                        planar = array.reshape((360*640,3)).transpose().copy().reshape((3,360,640))
                        owned = image.Image(640,360,image.RGBP888,alloc=image.ALLOC_REF,data=planar)
                        latencies['usb_rgb'].add(time.ticks_diff(time.ticks_us(),t))
                        if road:
                            t = time.ticks_us()
                            result['roadnet_center_logits'] = road.infer(owned)
                            latencies['roadnet'].add(time.ticks_diff(time.ticks_us(),t))
                            for stage, elapsed_us in road.timing_us.items():
                                latencies['roadnet_' + stage].add(elapsed_us)
                        if top:
                            t = time.ticks_us()
                            decoded = top.infer(owned)
                            result['topface_last_count'] = len(decoded['detections'])
                            latencies['topface'].add(time.ticks_diff(time.ticks_us(),t))
                            for stage in ('ai2d', 'kpu', 'decode'):
                                latencies['topface_' + stage].add(round(top.timing[stage + '_ms'] * 1000))
                        if road and top:
                            result['shared_inferences'] += 1
                            result['last_shared_uvc_frame'] = result['uvc_frames']
                        owned = planar = array = packed = None
                raw = None
            if sensor is not None:
                t = time.ticks_us()
                rgb = sensor.snapshot(timeout=300, auto_reboot=False)
                latencies['csi_capture'].add(time.ticks_diff(time.ticks_us(),t))
                if rgb is None:
                    result['csi_drops'] += 1
                    consecutive_csi += 1
                else:
                    consecutive_csi = 0
                    actual_size = (rgb.width(), rgb.height())
                    result['csi_frame_wh'] = list(actual_size)
                    if actual_size != (640,480):
                        raise ValueError('CSI snapshot size differs from configured mode')
                    result['csi_frames'] += 1
                    if mode in ('qr','both'):
                        t = time.ticks_us()
                        gray = rgb.copy().to_grayscale()
                        latencies['qr_grayscale'].add(time.ticks_diff(time.ticks_us(),t))
                        t = time.ticks_us()
                        codes = [code.payload() for code in gray.find_qrcodes()]
                        latencies['qr_decode'].add(time.ticks_diff(time.ticks_us(),t))
                        result['qr_frames'] += 1
                        if len(codes)>1:
                            result['qr_ambiguous'] += 1
                        elif len(codes)==1:
                            try:
                                parse_task(codes[0])
                            except ValueError:
                                result['qr_invalid'] += 1
                            else:
                                if expected_qr is not None:
                                    key = 'qr_match' if codes[0] == expected_qr else 'qr_wrong'
                                    result[key] += 1
                                else:
                                    result['qr_valid_unscored'] += 1
                        else:
                            result['qr_no_detection'] += 1
                        gray = None
                rgb = None
            if max(consecutive_uvc,consecutive_csi)>=10:
                raise RuntimeError('Ten consecutive camera acquisition failures')
            result['cycles'] += 1
            if result['cycles']%4 == 0:
                gc.collect()
            now = time.ticks_ms()
            if time.ticks_diff(now,last_memory_sample)>=1000:
                sample_memory(result, resources, 'active')
                last_memory_sample = now
            latencies['cycle'].add(time.ticks_diff(time.ticks_us(),cycle))
        if (wants_usb and not result['uvc_frames']) or (wants_csi and not result['csi_frames']):
            raise RuntimeError('A requested camera supplied zero frames')
        result['status'] = ('MEASURED_WITH_CAPTURE_MISSES'
                            if result['uvc_drops'] or result['csi_drops'] else 'MEASURED')
    except BaseException as error:
        result['status'] = 'ERROR'
        result['failure'] = {'type':type(error).__name__,'message':str(error)}
    finally:
        seconds = time.ticks_diff(time.ticks_ms(),started)/1000 if started is not None else 0
        try:
            raw = rgb = gray = owned = packed = array = planar = None
            gc.collect()
        except BaseException:
            pass
        operations = []
        if uvc_attempted:
            operations.append(('uvc',uvc.stop))
        if sensor_configured:
            operations.append(('csi',sensor.stop))
        if media_attempted:
            operations.append(('media',MediaManager.deinit))
        if road:
            operations.append(('roadnet',road.close))
        if top:
            operations.append(('topface',top.close))
        for label, action in operations:
            try:
                action()
            except BaseException as error:
                result['cleanup_errors'].append({'stage':label,'error':str(error)})
        if result['cleanup_errors']:
            result['status'] = 'CLEANUP_UNCONFIRMED'
        result['elapsed_seconds'] = seconds
        result['uvc_capture_fps'] = frame_rate(result['uvc_frames'],seconds) if wants_usb else None
        result['csi_capture_fps'] = frame_rate(result['csi_frames'],seconds) if wants_csi else None
        result['scheduler_cycles_fps'] = frame_rate(result['cycles'], seconds)
        result['uvc_attempts'] = latencies['uvc_capture'].count
        result['csi_attempts'] = latencies['csi_capture'].count
        result['latency'] = {name:record.report() for name,record in latencies.items()}
        try:
            sample_memory(result, resources, 'final')
        except BaseException as error:
            result['memory_sampling_failures'] += 1
            result['last_memory_sampling_failure'] = {'phase':'final', 'message':str(error)}
        if report_path is not None:
            with open(report_path,'w') as stream:
                json.dump(result,stream)
        print('ROADNET_SMOKE_END', mode, result['status'], report_path)
    return result
