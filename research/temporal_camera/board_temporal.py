"""Explicit CanMV bench for the experimental single-frame CameraXY KModel."""
import json
import math

from localization_runtime import CameraXYTracker


class BenchDecoder:
    def __init__(self, centers, settings, mode, clock, ticks_diff, latency):
        if mode not in ('raw', 'filtered'):
            raise ValueError('Expected raw or filtered benchmark mode')
        self.centers = centers
        self.mode = mode
        self.clock = clock
        self.ticks_diff = ticks_diff
        self.latency = latency
        self.last_tick = None
        self.elapsed_s = 0.
        self.states = {}
        self.tracker = CameraXYTracker(centers, **settings) if mode == 'filtered' else None

    def __call__(self, logits, map_centers, _confidence, _candidate_set_confidence):
        if map_centers != self.centers or len(logits) != len(self.centers):
            raise ValueError('Benchmark model output differs from its anchor contract')
        tick = self.clock()
        if self.last_tick is not None:
            self.elapsed_s += self.ticks_diff(tick, self.last_tick) / 1000000
        self.last_tick = tick
        if self.tracker is None:
            x, y = None, None
            if all(math.isfinite(value) for value in logits):
                best = max(range(len(logits)), key=lambda index: logits[index])
                x, y = self.centers[best]
            answer = dict(camera_x_mm=None, camera_y_mm=None, position_valid=False,
                          candidate_x_mm=x, candidate_y_mm=y)
        else:
            answer = self.tracker.update(logits, self.elapsed_s)
            self.latency.add(self.ticks_diff(self.clock(), tick))
            state = answer['state']
            self.states[state] = self.states.get(state, 0) + 1
        answer['status'] = 'unlocalized'
        return answer


def run(model_root, run_id, expected_usb_device, frames=120, mode='filtered'):
    """Collect native counters inside the explicitly loaded experimental directory."""
    import os
    import sys
    import time

    if sys.implementation.name != 'micropython':
        raise RuntimeError('Run on CanMV MicroPython with an idle OV2735')
    if not model_root.startswith('/sdcard/k230_temporal_v1_') or any(
        part in ('', '.', '..') for part in model_root.split('/')[1:]
    ):
        raise ValueError('Use a dedicated /sdcard/k230_temporal_v1_ directory')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in run_id):
        raise ValueError('Use a unique report ID containing only lowercase ASCII, digits, dash and underscore')

    import board_probe
    from board_helpers import Latency

    with open(model_root + '/deployment.json') as stream:
        deployment = json.load(stream)
    tracker_config = deployment['temporal_filter']
    if tracker_config.get('acceptance') is not None:
        raise ValueError('Real-camera confidence calibration is pending')
    centers = deployment['identity']['map_centers_mm']
    report_path = model_root + '/' + run_id + '.json'
    filter_report_path = report_path + '.filter.json'
    try:
        os.stat(filter_report_path)
    except OSError:
        pass
    else:
        raise ValueError('Filter benchmark report already exists')

    latency = Latency()
    decoder = BenchDecoder(centers, tracker_config, mode, time.ticks_us, time.ticks_diff, latency)
    original = board_probe.decode_map
    board_probe.decode_map = decoder
    try:
        report = board_probe.run(model_root, report_path, expected_usb_device, frames)
    finally:
        board_probe.decode_map = original
    report['mode'] = mode
    report['filter_latency'] = latency.report()
    report['temporal_states'] = decoder.states
    report['native_ai2d_pixel_equivalence'] = 'NOT RUN'
    report['real_camera_accuracy'] = 'NOT RUN'
    with open(filter_report_path + '.partial', 'w') as stream:
        stream.write(json.dumps(report))
    os.rename(filter_report_path + '.partial', filter_report_path)
    return report
