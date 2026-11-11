import copy
import json
from pathlib import Path
import unittest
from vision.ring import identify_ring
from k230.vision_adapter import VisionAdapter
from tools.replay_ring_candidates import replay

ROOT = Path(__file__).resolve().parents[1]


def gates():
    return {i: {'center': (x, 160), 'radius_px': 24, 'min_distance_gap_px': 1, 'min_support': 1}
            for i, x in enumerate((120, 320, 520), 1)}


class ReferenceRingTests(unittest.TestCase):
    def test_saved_native_failure_resolves_against_known_synthetic_layout(self):
        report = replay(ROOT/'evidence/board/native-fixtures-attempt02.json',
                        ROOT/'data/board_fixtures/synthetic-ring-layout.json')
        self.assertEqual(report['resolved'], 3)
        self.assertFalse(report['board_rerun'])
        for row, x in zip(report['results'], (120, 320, 520)):
            u, v = row['candidate']['center']
            self.assertLessEqual((u-x)**2+(v-160)**2, 8**2)
            self.assertIsNone(row['x_mm'])
            self.assertFalse(row['target_pose_valid'])

    def test_reference_not_inferred_without_verified_geometry(self):
        groups = [{'center': (120, 160), 'support': 2}]
        for cfg in ({}, {'center_gates': gates()}, {'verified': True},
                    {'verified': True, 'center_gates': None}):
            self.assertEqual(identify_ring(groups, 1, cfg), (None, 'ID_UNKNOWN'))

    def test_equally_plausible_candidates_still_refused(self):
        groups = [{'center': (118, 166), 'support': 2}, {'center': (122, 166), 'support': 2}]
        self.assertEqual(identify_ring(groups, 1, {'verified': True, 'center_gates': gates()}),
                         (None, 'ID_UNKNOWN'))

    def test_overlapping_numbered_regions_are_refused(self):
        config = gates()
        config[2]['center'] = (140, 160)
        self.assertEqual(identify_ring([{'center': (120, 160), 'support': 3}], 1,
                                      {'verified': True, 'center_gates': config}), (None, 'ID_UNKNOWN'))

    def test_invalid_reference_or_low_support_never_guesses(self):
        for field, value in [('center', (float('nan'), 160)), ('radius_px', -1),
                             ('min_distance_gap_px', 0), ('min_support', True), ('min_support', 3)]:
            cfg = gates()
            cfg[1][field] = value
            self.assertIsNone(identify_ring([{'center': (120, 160), 'support': 1}], 1,
                                           {'verified': True, 'center_gates': cfg})[0])

    def test_noise_outside_reference_and_missing_target_are_not_selected(self):
        self.assertIsNone(identify_ring([{'center': (180, 160), 'support': 10}], 1,
                                       {'verified': True, 'center_gates': gates()})[0])

    def test_adapter_keeps_source_binding_and_mm_refusal(self):
        from test_vision import config, Image, frame, PROFILE
        cfg = config()
        cfg['RING_CONFIG'] = {'PROCESS': {'roi': (0, 0, 640, 480), 'verified': True,
            'camera_id': 2, 'image_profile': dict(PROFILE), 'center_gates': gates()}}
        image = Image(circles=[(118, 166, 24), (128, 158, 30), (400, 200, 20)])
        value = VisionAdapter(cfg).detect('RING', frame(image), 1, 'PROCESS')
        self.assertEqual((value['u_px'], value['v_px']), (118, 166))
        self.assertEqual(value['status'], 'UNCALIBRATED')
        self.assertFalse(value['target_pose_valid'])
        self.assertIsNone(value['x_mm'])
        cfg['RING_CONFIG']['PROCESS']['image_profile']['hmirror'] = True
        value = VisionAdapter(cfg).detect('RING', frame(image), 1, 'PROCESS')
        self.assertEqual(value['reason'], 'ID_UNKNOWN')
        self.assertIsNone(value['target_id'])


if __name__ == '__main__':
    unittest.main()
