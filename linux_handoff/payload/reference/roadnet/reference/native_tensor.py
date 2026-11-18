"""Frozen-input KPU execution; independent of camera and field acceptance."""
import gc
import json
import math
import os
import struct
import time


def run(output_root, model_root, expected_input_sha, samples=10):
    import nncase_runtime as nn
    from ulab import numpy as np
    from k230.topface_adapter import digest
    from k230.topface_service import resources
    from board_smoke import Latency

    report_path = output_root + '/tensor-a.json'
    output_path = output_root + '/tensor-a.f32'
    for path in (report_path, output_path):
        try:
            os.stat(path)
        except OSError:
            pass
        else:
            raise ValueError('Preserve existing tensor evidence')
    result = {'scope': 'Exact held-out synthetic input on physical KPU; camera NOT_RUN',
              'requested_samples': samples, 'completed_samples': 0, 'cleanup_errors': [],
              'initial_memory': resources(), 'firmware': str(os.uname()), 'status': 'STARTED'}
    kpu = data = backing = input_tensor = output_tensor = raw = None
    timing = Latency()
    try:
        with open(model_root + '/identity.json') as stream:
            identity = json.load(stream)
        if digest(model_root + '/roadnet.kmodel') != identity['kmodel_sha256']:
            raise ValueError('KModel SHA mismatch')
        if digest(output_root + '/tensor-a.rgb') != expected_input_sha:
            raise ValueError('Input SHA mismatch')
        result['kmodel_sha256'] = identity['kmodel_sha256']
        result['input_sha256'] = expected_input_sha
        with open(output_root + '/tensor-a.rgb', 'rb') as stream:
            data = stream.read()
        if len(data) != 3*192*320:
            raise ValueError('Input length differs')
        backing = np.frombuffer(data, dtype=np.uint8).reshape((1, 3, 192, 320))
        input_tensor = nn.from_numpy(backing)
        kpu = nn.kpu()
        kpu.load_kmodel(model_root + '/roadnet.kmodel')
        result['runtime'] = str(nn.version())
        result['inputs'] = [str(kpu.inputs_desc(i)) for i in range(kpu.inputs_size())]
        result['outputs'] = [str(kpu.outputs_desc(i)) for i in range(kpu.outputs_size())]
        kpu.set_input_tensor(0, input_tensor)
        for index in range(samples):
            os.exitpoint()
            start = time.ticks_us()
            kpu.run()
            timing.add(time.ticks_diff(time.ticks_us(), start))
            result['completed_samples'] += 1
        output_tensor = kpu.get_output_tensor(0)
        raw = output_tensor.to_numpy()
        result['output_shape'] = list(raw.shape)
        result['output_dtype'] = str(raw.dtype)
        if list(raw.shape) != identity['output_logits_nchw']:
            raise ValueError('Output shape differs')
        finite = 0
        with open(output_path, 'wb') as stream:
            for channel in range(7):
                for y in range(192):
                    row = [float(raw[0, channel, y, x]) for x in range(320)]
                    finite += sum(1 for value in row if math.isfinite(value))
                    stream.write(struct.pack('<320f', *row))
                gc.collect()
        result['finite_values'] = finite
        result['expected_values'] = 7*192*320
        result['output_bytes'] = os.stat(output_path)[6]
        result['output_sha256'] = digest(output_path)
        result['active_memory'] = resources()
        result['status'] = 'NATIVE_TENSOR_COMPLETE' if finite == 7*192*320 else 'NONFINITE_OUTPUT'
    except BaseException as error:
        result['status'] = 'ERROR'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        raw = output_tensor = kpu = input_tensor = backing = data = None
        gc.collect()
        try:
            nn.shrink_memory_pool()
        except BaseException as error:
            result['cleanup_errors'].append(str(error))
        result['kpu_latency'] = timing.report()
        result['final_memory'] = resources()
        with open(report_path, 'w') as stream:
            json.dump(result, stream)
        print('ROADNET_NATIVE_TENSOR:' + json.dumps(result))
    return result


def shared(output_root, model_root, top_root, samples=20):
    """Qualify both AI2D/KPU paths on one owned synthetic frame, without cameras."""
    import image
    from ulab import numpy as np
    from k230.topface_adapter import TopFaceBackend
    from k230.topface_service import resources
    from board_smoke import RoadNetTrial, Latency

    report_path = output_root + '/shared-a.json'
    try:
        os.stat(report_path)
    except OSError:
        pass
    else:
        raise ValueError('Preserve existing shared-frame evidence')
    result = {'scope': 'Native AI2D and two KPU models on one owned constant RGB frame; cameras NOT_RUN',
              'input': {'shape': [3, 360, 640], 'rgb': [90, 140, 190], 'source': 'synthetic_constant'},
              'requested_cycles': samples, 'shared_inferences': 0, 'cleanup_errors': [],
              'initial_memory': resources(), 'status': 'STARTED'}
    road = top = frame = backing = None
    names = ('cycle', 'roadnet', 'roadnet_ai2d', 'roadnet_kpu', 'roadnet_readback',
             'topface', 'topface_ai2d', 'topface_kpu', 'topface_decode')
    timing = {name: Latency() for name in names}
    try:
        road = RoadNetTrial(model_root)
        top = TopFaceBackend(top_root)
        result['roadnet_sha256'] = road.identity['kmodel_sha256']
        result['topface_sha256'] = top.identity['kmodel_sha256']
        backing = np.zeros((3, 360, 640), dtype=np.uint8)
        for channel, value in enumerate((90, 140, 190)):
            backing[channel, :, :] = value
        frame = image.Image(640, 360, image.RGBP888, alloc=image.ALLOC_REF, data=backing)
        start_ms = time.ticks_ms()
        for index in range(samples):
            os.exitpoint()
            cycle = time.ticks_us()
            start = time.ticks_us()
            result['roadnet_center_logits'] = road.infer(frame)
            timing['roadnet'].add(time.ticks_diff(time.ticks_us(), start))
            for name, elapsed in road.timing_us.items():
                timing['roadnet_' + name].add(elapsed)
            start = time.ticks_us()
            decoded = top.infer(frame)
            timing['topface'].add(time.ticks_diff(time.ticks_us(), start))
            for name in ('ai2d', 'kpu', 'decode'):
                timing['topface_' + name].add(round(top.timing[name + '_ms'] * 1000))
            result['topface_last_count'] = len(decoded['detections'])
            result['shared_inferences'] += 1
            if (index + 1) % 4 == 0:
                gc.collect()
            timing['cycle'].add(time.ticks_diff(time.ticks_us(), cycle))
        result['elapsed_seconds'] = time.ticks_diff(time.ticks_ms(), start_ms) / 1000
        result['active_memory'] = resources()
        result['status'] = 'NATIVE_SHARED_SYNTHETIC_COMPLETE'
    except BaseException as error:
        result['status'] = 'ERROR'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        frame = backing = None
        for label, model in (('roadnet', road), ('topface', top)):
            if model is not None:
                try:
                    model.close()
                except BaseException as error:
                    result['cleanup_errors'].append({'stage': label, 'message': str(error)})
        result['latency'] = {name: value.report() for name, value in timing.items()}
        gc.collect()
        result['final_memory'] = resources()
        with open(report_path, 'w') as stream:
            json.dump(result, stream)
        print('ROADNET_NATIVE_SHARED:' + json.dumps(result))
    return result
