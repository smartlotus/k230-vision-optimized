"""Bounded CanMV single/dual/triple acquisition checks, no motors or UART."""
import sys
if sys.implementation.name != 'micropython':
    raise SystemExit('Run with CanMV on the connected K230.')
import os
import time
import gc
import json
ROOT = '/sdcard/transport_mvp'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from k230.camera_manager import CameraManager


def acquire(ids, rounds, save_samples=False):
    cfg = {'CAMERA_IDS': ids, 'DESKTOP_SINGLE_CAMERA': False, 'PREVIEW': False,
           'WIDTH': 640, 'HEIGHT': 480, 'SENSOR_WIDTH': 1920, 'SENSOR_HEIGHT': 1080,
           'SENSOR_FPS': 30, 'FIRST_FRAME_TIMEOUT_MS': 2000, 'SNAPSHOT_TIMEOUT_MS': 100}
    camera = CameraManager(cfg)
    record = {'camera_ids': list(ids), 'expected_rounds': rounds, 'finished': False,
              'source': 'REAL_CSI_FRAMES', 'motors_enabled': False, 'uart_enabled': False,
              'counts': {str(i): 0 for i in ids}, 'capture_ms': {str(i): [] for i in ids},
              'first_frame': {}, 'last_frame': {}, 'samples': {}, 'errors': []}
    try:
        print('K230_MULTICAM_PHASE:start_'+str(ids))
        camera.start()
        for index in range(rounds):
            os.exitpoint()
            for camera_id in ids:
                before = time.ticks_ms()
                frame = camera.get_frame(camera_id)
                if frame is None or frame['camera_id'] != camera_id:
                    raise RuntimeError('Missing or wrong-source frame '+str(camera_id))
                key = str(camera_id)
                record['capture_ms'][key].append(time.ticks_diff(time.ticks_ms(), before))
                record['counts'][key] += 1
                metadata = {k: frame[k] for k in ('camera_id', 'frame_id', 'timestamp_ms', 'image_profile', 'is_simulated')}
                if key not in record['first_frame']:
                    record['first_frame'][key] = metadata
                record['last_frame'][key] = metadata
                if save_samples and index == rounds-1:
                    path = ROOT+'/threecam_csi'+key+'.jpg'
                    frame['image'].save(path, quality=85)
                    record['samples'][key] = path
            time.sleep_ms(10)
        record['finished'] = True
    except Exception as error:
        record['errors'].append(str(error))
        sys.print_exception(error)
    finally:
        print('K230_MULTICAM_PHASE:close_'+str(ids))
        camera.close()
        record['cleanup_errors'] = camera.close_errors
        gc.collect()
        record['heap_free_bytes'] = gc.mem_free()
        record['mean_capture_ms'] = {key: sum(values)/len(values) if values else None
                                     for key, values in record['capture_ms'].items()}
        record['pass'] = (record['finished'] and not record['errors'] and not record['cleanup_errors']
                          and all(count == rounds for count in record['counts'].values()))
    print('K230_MULTICAM_STAGE:'+json.dumps({k: record[k] for k in ('camera_ids', 'counts', 'pass', 'errors', 'cleanup_errors')}))
    return record


report = {'test': 'SINGLE_DUAL_TRIPLE_CSI_ACQUISITION', 'stages': [], 'finished': False,
          'motors_enabled': False, 'uart_enabled': False}
try:
    for camera_id in (0, 1, 2):
        report['stages'].append(acquire((camera_id,), 6))
    if all(stage['pass'] for stage in report['stages']):
        report['stages'].append(acquire((2, 0), 6))
    if len(report['stages']) == 4 and report['stages'][-1]['pass']:
        report['stages'].append(acquire((2, 0, 1), 20, save_samples=True))
    report['finished'] = True
finally:
    report['pass'] = len(report['stages']) == 5 and all(stage['pass'] for stage in report['stages'])
    with open(ROOT+'/threecam_qualification.json', 'w') as output:
        output.write(json.dumps(report))
    print('K230_MULTICAM_RESULT:'+json.dumps({'finished': report['finished'], 'pass': report['pass'], 'stages': len(report['stages'])}))
