import json
from pathlib import Path
import tempfile
import unittest

from cloud_store import sha
from evaluate_redetection import load_sequence_frames
from scene_config import camera_partition
from sequence_sweep import build_plan, planned_config


class SequenceSweepTests(unittest.TestCase):
    def test_all_conditions_keep_the_test_partition_and_six_colors(self):
        plan = build_plan()
        self.assertEqual(plan['count'], 1872)
        self.assertEqual(len(plan['sequences']), 156)
        self.assertEqual({camera[0] for camera in plan['cameras']}, {100, 175, 250, 325, 400})
        self.assertEqual({camera[1] for camera in plan['cameras']}, {80, 85, 90, 95, 100})
        seeds = set()
        for group, sequence in enumerate(plan['sequences']):
            first = planned_config(plan, group * 12)
            last = planned_config(plan, group * 12 + 11)
            for config in (first, last):
                self.assertEqual(camera_partition(config['camera_height_above_top_mm'],
                    config['plane_angle_deg'], config['hfov_deg'])[1], 'synthetic_test')
                self.assertEqual(config['objects'][0]['color_id'], sequence['color_id'])
                self.assertNotIn(config['random_seed'], seeds)
                seeds.add(config['random_seed'])
            self.assertEqual(first['light_position'], last['light_position'])
            self.assertEqual(first['background_rgb'], last['background_rgb'])
            if sequence['event'].startswith('camera_vertical_'):
                self.assertEqual(first['objects'][0]['position_m'], [0., 0., 0.])
                self.assertEqual(first['objects'], last['objects'])
                self.assertEqual((first['plane_angle_deg'], first['hfov_deg']), (90, 65))
                self.assertNotEqual(first['camera_height_above_top_mm'], last['camera_height_above_top_mm'])
            else:
                self.assertEqual(first['camera_pose_world_from_cv'], last['camera_pose_world_from_cv'])
        self.assertEqual(sorted(plan['vertical_heights_mm']), plan['vertical_heights_mm'])
        self.assertEqual(len(set(plan['vertical_heights_mm'])), 12)

    def fixture(self, root):
        rows = []
        for frame in range(12):
            directory = root / f'scene-{frame:05d}'
            directory.mkdir()
            (directory / 'rgb.png').write_bytes(b'audited image fixture')
            hidden = 4 <= frame < 8
            label = {'split': 'synthetic_test', 'instances': [
                {'color_id': 1, 'visible_area_px': 0 if hidden else 50}],
                'expected_visible': not hidden, 'sequence_group_id': 'one', 'sequence_frame': frame,
                'sequence_event': 'occlusion', 'recovery_window_start': 8, 'sequence_length': 12}
            (directory / 'labels.json').write_text(json.dumps(label), encoding='utf8')
            rows.append({'scene': directory.name, 'rgb_sha256': sha(directory / 'rgb.png'),
                         'label_sha256': sha(directory / 'labels.json')})
        return {'scenes': rows}

    def test_missing_or_duplicate_frames_are_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index = self.fixture(root)
            self.assertEqual(len(load_sequence_frames(root, index)), 12)
            for rows in (index['scenes'][:-1], index['scenes'] + index['scenes'][:1]):
                with self.assertRaisesRegex(ValueError, 'missing, duplicate'):
                    load_sequence_frames(root, {'scenes': rows})

    def test_changed_color_and_incorrect_hidden_schedule_are_rejected(self):
        for mutation in ('color', 'hidden'):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                index = self.fixture(root)
                row = index['scenes'][5]
                path = root / row['scene'] / 'labels.json'
                label = json.loads(path.read_text(encoding='utf8'))
                if mutation == 'color':
                    label['instances'][0]['color_id'] = 2
                else:
                    label['expected_visible'] = True
                    label['instances'][0]['visible_area_px'] = 50
                path.write_text(json.dumps(label), encoding='utf8')
                row['label_sha256'] = sha(path)
                with self.assertRaises(ValueError):
                    load_sequence_frames(root, index)


if __name__ == '__main__':
    unittest.main()
