import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from replay_camera import load_trajectories, make_model_input, replay


class FixedScores:
    def __init__(self):
        self.inputs = []

    def run(self, _, inputs):
        rgb = inputs['rgb_window']
        self.inputs.append(rgb.copy())
        recovered = float(rgb[0, -3, 4, 0]) > .3
        return [np.array([[-20., 20.] if recovered else [20., -20.]], np.float32)]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i, color in enumerate((10, None, None, None, None, None, 100, 120)):
            image = None
            if color is not None:
                image = f'{i}.png'
                Image.new('RGB', (160, 90), (color, color, color)).save(self.root / image)
            self.rows.append({'trajectory_id': 'a', 'frame_index': i, 'timestamp_s': i / 10, 'image_path': image})
        self.manifest = self.write('frames.jsonl', self.rows)

    def write(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def routes(self):
        return {'tcn4': {'net': FixedScores(), 'config': {'grid_size': 24}, 'sha256': 'test-fixture'}}

    def test_drop_recovery_and_causal_window(self):
        routes = self.routes()
        output = self.root / 'out.json'
        report = replay(self.manifest, output, routes, [(1050, 1050), (-850, -550)], {}, 'simulated')
        self.assertEqual(json.loads(output.read_text()), report)
        summary = report['routes']['tcn4']['summary']
        for key, value in {'capture_attempts': 8, 'observed_frames': 3, 'missing_captures': 5,
                           'lost_updates': 1, 'global_recovery_events': 1, 'history_resets': 1}.items():
            self.assertEqual(summary[key], value)
        samples = routes['tcn4']['net'].inputs
        np.testing.assert_allclose(samples[0][0, ::3, 4, 0], [10 / 255] * 4)
        np.testing.assert_allclose(samples[1][0, ::3, 4, 0], [100 / 255] * 4)
        np.testing.assert_allclose(samples[2][0, ::3, 4, 0], [100 / 255] * 3 + [120 / 255])
        frames = report['routes']['tcn4']['trajectories']['a']['frames']
        self.assertIsNone(frames[5]['filtered_camera_xy_candidate_mm'])
        self.assertEqual(frames[6]['filtered_camera_xy_candidate_mm'], [-850, -550])
        self.assertTrue(all(not row['position_valid'] for row in frames))
        self.assertEqual(summary['surveyed_filtered']['status'], 'NOT RUN')
        with self.assertRaises(FileExistsError):
            replay(self.manifest, output, routes, [(1050, 1050), (-850, -550)], {}, 'simulated')

    def test_scoring_metadata_does_not_enter_inference(self):
        original = self.routes()
        original_report = replay(self.manifest, self.root / 'a.json', original,
                                 [(1050, 1050), (-850, -550)], {}, 'simulated')
        labeled = copy.deepcopy(self.rows)
        labeled[0].update(start_zone=2, camera_xy_target={'camera_x_mm': -1000, 'camera_y_mm': -1000},
                         measurement={'reference': 'DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY',
                                      'uncertainty_mm': 5, 'method': 'unit fixture', 'record_id': 'a'})
        labeled[6]['first_landmark'] = True
        changed = self.routes()
        changed_report = replay(self.write('labeled.jsonl', labeled), self.root / 'b.json', changed,
                                [(1050, 1050), (-850, -550)], {}, 'simulated')
        before = original_report['routes']['tcn4']['trajectories']['a']['frames']
        after = changed_report['routes']['tcn4']['trajectories']['a']['frames']
        for a, b in zip(before, after):
            for key in ('raw_camera_xy_candidate_mm', 'filtered_camera_xy_candidate_mm', 'state',
                        'posterior_local_mass', 'blue_region_posterior'):
                self.assertEqual(a[key], b[key])
        for a, b in zip(original['tcn4']['net'].inputs, changed['tcn4']['net'].inputs):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(changed_report['routes']['tcn4']['summary']['surveyed_filtered']['n'], 1)

    def test_reject_timestamp_path_and_extra_supervision(self):
        broken = copy.deepcopy(self.rows)
        broken[1]['timestamp_s'] = broken[0]['timestamp_s']
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            load_trajectories(self.write('bad-time.jsonl', broken))
        broken = copy.deepcopy(self.rows)
        broken[0]['image_path'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'relative image'):
            load_trajectories(self.write('bad-path.jsonl', broken))
        broken = copy.deepcopy(self.rows)
        broken[0]['camera_xy_target'] = {'camera_x_mm': 0, 'camera_y_mm': 0, 'camera_yaw_deg': 0}
        with self.assertRaisesRegex(ValueError, 'Only camera'):
            load_trajectories(self.write('bad-target.jsonl', broken))

    def test_bad_hash_and_missing_capture_labels_are_rejected(self):
        broken = copy.deepcopy(self.rows)
        broken[0]['image_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_trajectories(self.write('bad-hash.jsonl', broken))
        broken = copy.deepcopy(self.rows)
        broken[1]['first_landmark'] = True
        with self.assertRaisesRegex(ValueError, 'missing capture'):
            load_trajectories(self.write('bad-missing.jsonl', broken))

    def test_sensor_resize_and_storage_layout(self):
        image = self.root / 'sensor.png'
        Image.new('RGB', (640, 360), (254, 1, 113)).save(image)
        model_input, policy = make_model_input(image)
        self.assertEqual(policy, 'RGB640X360_AI2D_EQUIVALENT')
        self.assertEqual(model_input.shape, (3, 96, 160))
        np.testing.assert_array_equal(model_input[:, 3, 0], [254, 1, 113])
        self.assertTrue(np.all(model_input[:, :3] == 114))


if __name__ == '__main__':
    unittest.main()
