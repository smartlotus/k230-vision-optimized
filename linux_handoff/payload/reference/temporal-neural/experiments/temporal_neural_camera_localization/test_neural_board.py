"""Exercise the generated CanMV entry with bounded fake capture and KPU failures."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as S
import time
import unittest
from unittest.mock import patch

import numpy as np

import board_support
from neural_window import FrameWindow
from package_neural_trial import bench_source


class NeuralBoardTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / 'native.json'
        self.center = [[float(x), float(y)] for y in range(-1175, 1200, 50) for x in range(-1175, 1200, 50)]
        payload = b'mock-kmodel-only'
        (self.root / 'camera_xy.kmodel').write_bytes(payload)
        unknown = dict(temperature=1., threshold=None)
        metadata = dict(identity=dict(map_centers_mm=self.center, input_rgb_uint8_nchw=[1, 24, 96, 160],
                                      confidence=unknown, candidate_set_confidence=unknown, cell_mm=50),
                        neural_window=dict(length=8, sample_us=230000, max_gap_us=730405),
                        kmodel=dict(kmodel_sha256=hashlib.sha256(payload).hexdigest()))
        (self.root / 'deployment.json').write_text(json.dumps(metadata))
        source = Path(__file__).resolve().parents[1] / 'camera_xy_localization_v2/board_probe.py'
        self.namespace = {}
        exec(bench_source(source.read_text(encoding='utf8')), self.namespace)
        self.cleaned, self.inputs = [], []
        self.ticks = 0

    def invoke(self, snapshots, failure=None, cleanup_failure=False):
        captures = iter(snapshots)

        def snapshot(timeout):
            self.ticks, color = next(captures)
            if color is None:
                return None
            value = np.full((360, 640, 3), color, dtype=np.uint8)
            return S(to_rgb888=lambda **kw: S(to_numpy_ref=lambda: value))

        def close_uvc():
            self.cleaned.append('uvc')
            if cleanup_failure:
                raise OSError('mock UVC close failed')

        uvc = S(probe=lambda: (True, 'mock'), FOURCC_MJPEG=0, video_mode=lambda *args: args,
                select_video_mode=lambda mode: (True, S(width=1280, height=720, fourcc=0, fps=30)),
                start=lambda **kw: True, snapshot=snapshot, stop=close_uvc)

        def run_kpu():
            if failure:
                raise RuntimeError(failure)
            self.inputs.append(kpu.tensor[0, ::3, 0, 0].tolist())

        kpu = S(load_kmodel=lambda path: None, inputs_size=lambda: 1, outputs_size=lambda: 1,
                set_input_tensor=lambda index, tensor: setattr(kpu, 'tensor', tensor), run=run_kpu,
                get_output_tensor=lambda index: S(to_numpy=lambda: np.zeros((1, 2304), dtype=np.float32)))
        builder = S(run=lambda source, target: target.fill(source[0, 0, 0, 0]))
        ai2d = S(set_dtype=lambda *args: None, set_pad_param=lambda *args: None,
                 set_resize_param=lambda *args: None, build=lambda *args: builder)
        runtime = S(kpu=lambda: kpu, ai2d=lambda: ai2d, from_numpy=lambda value: value,
                    ai2d_format=S(NCHW_FMT=0), interp_method=S(tf_bilinear=0), interp_mode=S(half_pixel=0))
        modules = {'media': ModuleType('media'), 'media.uvc': S(UVC=uvc),
                   'media.media': S(MediaManager=S(init=lambda: None, deinit=lambda: self.cleaned.append('media'))),
                   'nncase_runtime': runtime, 'ulab': S(numpy=np)}
        with patch.dict(sys.modules, modules), patch.object(os, 'exitpoint', lambda: None, create=True), \
                patch.object(os, 'uname', lambda: 'mock', create=True), \
                patch.object(time, 'ticks_us', lambda: self.ticks, create=True), \
                patch.object(time, 'ticks_diff', lambda a, b: a - b, create=True), \
                patch.object(board_support, 'resources', lambda: {'python_heap_free': 123456}):
            return self.namespace['run'](str(self.root), str(self.report), 'mock', len(snapshots))

    def test_trace_alignment_decimation_gap_reset_and_cleanup(self):
        result = self.invoke([(33000, 1), (66000, 2), (300000, None), (450000, 3), (1500000, 4)])
        self.assertEqual((result['frames'], result['dropped'], result['capture_decimated'], result['resets']),
                         (3, 1, 1, 1))
        self.assertEqual(self.inputs, [[1] * 8, [1] * 7 + [3], [4] * 8])
        records = [json.loads(line) for line in Path(result['frames_path']).read_text().splitlines()]
        self.assertEqual([record['capture_index'] for record in records], [0, 3, 4])
        self.assertEqual(records[1]['window_timestamps_us'], [33000] * 7 + [450000])
        self.assertEqual(records[2]['window_timestamps_us'], [1500000] * 8)
        self.assertIsNone(records[2]['prediction']['camera_x_mm'])
        self.assertEqual(self.cleaned, ['uvc', 'media'])

    def test_kpu_exception_closes_media_and_writes_partial_run_report(self):
        with self.assertRaisesRegex(RuntimeError, 'mock failure'):
            self.invoke([(100000, 1)], failure='mock failure')
        report = json.loads(self.report.read_text())
        self.assertEqual(report['status'], 'RUN_FAILED')
        self.assertEqual(self.cleaned, ['uvc', 'media'])
        self.assertTrue(Path(report['frames_path']).is_file())

    def test_sixteen_frame_board_entry_and_trace(self):
        path = self.root / 'deployment.json'
        deployment = json.loads(path.read_text())
        deployment['neural_window']['length'] = 16
        deployment['identity']['input_rgb_uint8_nchw'] = [1, 48, 96, 160]
        path.write_text(json.dumps(deployment))
        result = self.invoke([(100000, 7), (400000, 8), (700000, 9)])
        self.assertEqual(result['window'], 16)
        self.assertEqual(self.inputs, [[7] * 16, [7] * 15 + [8], [7] * 14 + [8, 9]])
        records = [json.loads(line) for line in Path(result['frames_path']).read_text().splitlines()]
        self.assertEqual(records[-1]['window_timestamps_us'], [100000] * 14 + [400000, 700000])

    def test_continuous_missing_capture_and_cleanup_failure_recorded(self):
        with self.assertRaisesRegex(RuntimeError, 'No valid OV2735 frames'):
            self.invoke([(i * 100000, None) for i in range(10)], cleanup_failure=True)
        report = json.loads(self.report.read_text())
        self.assertEqual(report['status'], 'CLEANUP_UNCONFIRMED')
        self.assertEqual(report['dropped'], 10)
        self.assertEqual(report['cleanup_errors'], [['uvc', 'mock UVC close failed']])
        self.assertEqual(self.cleaned, ['uvc', 'media'])

    def test_existing_report_and_trace_preserved(self):
        for path in (self.report, Path(str(self.report) + '.frames.jsonl')):
            path.write_text('existing')
            with self.assertRaises(ValueError):
                self.invoke([(100000, 1)])
            self.assertEqual(path.read_text(), 'existing')
            path.unlink()
        self.assertFalse(self.cleaned)

    def test_ten_thousand_updates_reuse_fixed_buffers(self):
        buffer = FrameWindow(np, 8, 230000, 730405, lambda a, b: a - b)
        pointers = (buffer.history.ctypes.data, buffer.tensor.ctypes.data)
        for i in range(10000):
            buffer.append(np.full((1, 3, 96, 160), i % 251, dtype=np.uint8), i * 250000)
        gc.collect()
        self.assertEqual((buffer.history.ctypes.data, buffer.tensor.ctypes.data), pointers)
        self.assertEqual((buffer.count, len(buffer.tensor_timestamps_us), buffer.resets), (8, 8, 0))
        self.assertEqual(buffer.tensor[0, ::3, 0, 0].tolist(), [i % 251 for i in range(9992, 10000)])
