"""Bounded real-camera lifecycle and in-process control-byte test on CanMV.

No FPIOA or real UART is initialized. Synthetic controls exercise the real
event loop/codec; captured observations remain real and uncalibrated.
"""
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
from k230.main import run_loop
from k230.camera_manager import CameraManager
from k230.vision_adapter import VisionAdapter
from k230.uart_adapter import UartAdapter
from k230 import config
from ports.protocol import StreamParser, encode_control, encode_observation, encode_ack


class TestControlBytes:
    enabled = False
    reason = 'IN_PROCESS_BYTES_ONLY_NO_GPIO_OR_UART'

    def __init__(self):
        self.index = 0
        self.tx_seq = 0
        self.parser = StreamParser()
        self.wire = []
        self.observations = []
        self.closed = False
        self.tx_buffer = b''

    def poll(self):
        self.index += 1
        commands = {1: (1, 'MODE', 'QR', None, 'QR'), 2: (2, 'PING'),
                    4: (3, 'MODE', 'COLOR', 1, 'RAW'),
                    5: (4, 'MODE', 'RING', 1, 'PROCESS'),
                    6: (5, 'MODE', 'LANE'),
                    7: (6, 'MODE', 'TOP', 1, 'BUFFER'),
                    8: (7, 'MODE', 'IDLE'), 9: (8, 'STOP')}
        values = commands.get(self.index)
        if not values:
            return []
        payload = encode_control(*values)
        if self.index == 2:
            bad = encode_control(60000, 'MODE', 'TOP', 6, 'BUFFER')
            payload = bad[:-3] + (b'00\n' if bad[-3:] != b'00\n' else b'FF\n') + payload
        if self.index == 5:
            payload = encode_control(3, 'MODE', 'COLOR', 6, 'RAW') + payload
        return self.parser.feed(payload, time.ticks_ms())

    def send_ack(self, seq, status, reason=''):
        self.wire.append(encode_ack(self.tx_seq, status, reason, command_seq=seq))
        self.tx_seq += 1
        return True

    def send_observation(self, observation):
        self.observations.append(observation)
        self.wire.append(encode_observation(self.tx_seq, observation))
        self.tx_seq += 1
        return True

    def heartbeat(self, now_ms):
        return self.send_ack(0, 'PONG', 'IN_PROCESS_HEARTBEAT')

    def drain(self, max_writes=16):
        return True

    def close(self):
        self.closed = True


report = {'test': 'RECONNECT_REAL_CAMERA_WITH_IN_PROCESS_CONTROL',
          'source': 'REAL_CSI2_FRAMES_AND_SYNTHETIC_CONTROL_BYTES',
          'control_transport': 'IN_PROCESS_TEST_NOT_ELECTRICAL_UART',
          'motors_enabled': False, 'physical_uart_enabled': False,
          'finished': False, 'phases': [], 'errors': []}
config.PREVIEW = False
try:
    print('KMVP_RECONNECT_PHASE:control_loop')
    control = TestControlBytes()
    metrics = run_loop(CameraManager(config), VisionAdapter(config), control, config,
                       exitpoint=os.exitpoint, sleep_ms=time.sleep_ms, max_iterations=12)
    decoded = StreamParser().feed(b''.join(control.wire), time.ticks_ms())
    report['phases'].append({'phase': 'CONTROL_STOP', 'metrics': metrics,
                            'observations': control.observations,
                            'decoded_frames': len(decoded), 'encoded_frames': len(control.wire),
                            'acks': [row for row in decoded if row['type'] == 'A'],
                            'rejected_control_frames': control.parser.rejected,
                            'closed': control.closed})
    print('KMVP_RECONNECT_PHASE:restart_same_process')
    metrics = run_loop(CameraManager(config), VisionAdapter(config), UartAdapter(config.UART), config,
                       exitpoint=os.exitpoint, sleep_ms=time.sleep_ms, max_iterations=4)
    report['phases'].append({'phase': 'RESTART_SAME_PROCESS', 'metrics': metrics})
    print('KMVP_RECONNECT_PHASE:capture_file')
    camera = CameraManager(config)
    try:
        camera.start()
        for _ in range(4):
            os.exitpoint()
            frame = camera.get_frame(2)
        frame['image'].save(ROOT+'/reconnect-live.jpg', quality=85)
        report['capture_file'] = {'path': ROOT+'/reconnect-live.jpg', 'camera_id': 2,
                                  'frame_id': frame['frame_id'], 'image_profile': frame['image_profile']}
    finally:
        camera.close()
        report['capture_cleanup_errors'] = camera.close_errors
    report['finished'] = True
except Exception as error:
    report['errors'].append(str(error))
    sys.print_exception(error)
finally:
    gc.collect()
    report['heap_free_bytes'] = gc.mem_free()
    with open(ROOT+'/reconnect_lifecycle.json', 'w') as output:
        output.write(json.dumps(report))
    print('KMVP_RECONNECT_RESULT:'+json.dumps({'finished': report['finished'], 'errors': report['errors'],
                                             'phases_completed': len(report['phases'])}))
