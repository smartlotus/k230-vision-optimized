"""Bounded 120-iteration CSI2 observation run; no motion or physical UART."""
import sys
if sys.implementation.name != 'micropython':
    raise SystemExit('Run explicitly in CanMV on the K230.')
import os
import time
import json
import gc
ROOT = '/sdcard/transport_mvp'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from k230 import config
from k230.main import run_loop
from k230.camera_manager import CameraManager
from k230.vision_adapter import VisionAdapter
from k230.uart_adapter import UartAdapter


class CountedCamera(CameraManager):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.returned_frames = []
        self.unavailable_frames = 0

    def get_frame(self, camera_id):
        frame = super().get_frame(camera_id)
        if frame is None:
            self.unavailable_frames += 1
        else:
            self.returned_frames.append((frame['camera_id'], frame['frame_id'], frame['timestamp_ms']))
        return frame


config.PREVIEW = False
camera = CountedCamera(config)
report = {'test': 'BOUNDED_120_ITERATION_CSI2_RUN', 'motors_enabled': False,
          'physical_uart_enabled': False, 'expected_frames': 120, 'finished': False, 'errors': []}
try:
    metrics = run_loop(camera, VisionAdapter(config), UartAdapter(config.UART), config,
                       exitpoint=os.exitpoint, sleep_ms=time.sleep_ms, max_iterations=120)
    report['metrics'] = metrics
    report['successful_snapshot_count'] = len(camera.returned_frames)
    report['unavailable_frames'] = camera.unavailable_frames
    report['first_frame'] = camera.returned_frames[0] if camera.returned_frames else None
    report['last_frame'] = camera.returned_frames[-1] if camera.returned_frames else None
    report['strictly_increasing_ids'] = all(a[1] < b[1] for a, b in zip(camera.returned_frames, camera.returned_frames[1:]))
    report['finished'] = True
    report['pass'] = (len(camera.returned_frames) == 120 and camera.unavailable_frames == 0
                      and report['strictly_increasing_ids'] and not metrics['camera_cleanup_errors'])
except Exception as error:
    report['errors'].append(str(error))
    report['pass'] = False
    sys.print_exception(error)
finally:
    gc.collect()
    report['heap_free_bytes'] = gc.mem_free()
    with open(ROOT+'/capture_soak.json', 'w') as output:
        output.write(json.dumps(report))
    print('KMVP_CAPTURE_SOAK:'+json.dumps(report))
