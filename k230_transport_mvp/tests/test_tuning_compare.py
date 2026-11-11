import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from k230.tuning.schema import defaults, clone
from tools.board_tuning_green_compare import comparison_profiles, validate_request, save_report


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.profile = defaults()
        self.profile['contexts']['TOP_BUFFER']['thresholds']['4'] = [44, 57, -51, -33, 19, 38]
        self.request = dict(attempt='20261103-01', scene_id='new-scene', profile=self.profile['profile_id'], profile_file='scene03.json',
                            face_region=[20, 30, 40, 50], operator_ready=True)

    def test_candidates_preserve_other_colors_geometry_and_source(self):
        original = clone(self.profile)
        profiles = comparison_profiles(self.profile)
        for key, l, b in [('L41_B17', 41, 17), ('L40_B15', 40, 15)]:
            expected = clone(original)
            expected['profile_id'] += '-' + key
            expected['revision'] += 1
            expected['contexts']['TOP_BUFFER']['thresholds']['4'][0] = l
            expected['contexts']['TOP_BUFFER']['thresholds']['4'][4] = b
            self.assertEqual(profiles[key], expected)
        self.assertEqual(self.profile, original)
        measured = clone(profiles['L41_B17'])
        measured['profile_id'] = original['profile_id'] + '-L41_A30_B17'
        measured['contexts']['TOP_BUFFER']['thresholds']['4'][3] = -30
        self.assertEqual(profiles['L41_A30_B17'], measured)

    def test_stale_profile_no_readiness_or_outside_region_rejected(self):
        validate_request(self.request, self.profile)
        for edit in [dict(profile='old'), dict(operator_ready=False), dict(face_region=[630, 0, 20, 10]),
                     dict(attempt='../old'), dict(profile_file='../old.json'), dict(face_region=[True, 0, 10, 10])]:
            with self.assertRaises(ValueError):
                validate_request(dict(self.request, **edit), self.profile)

    def test_existing_partial_or_final_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'result.json')
            save_report({'frames': 100}, path)
            with self.assertRaises(OSError):
                save_report({'frames': 0}, path)
            self.assertEqual(json.loads(Path(path).read_text()), {'frames': 100})
            other = str(Path(root) / 'partial.json')
            Path(other + '.tmp').write_text('partial')
            with self.assertRaises(OSError):
                save_report({}, other)
            self.assertEqual(Path(other + '.tmp').read_text(), 'partial')

    def test_rename_failure_leaves_partial_and_never_returns_success(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'result.json')
            with patch('tools.board_tuning_green_compare.os.rename', side_effect=OSError('SD failure')):
                with self.assertRaises(OSError):
                    save_report({'frames': 100}, path)
            self.assertFalse(Path(path).exists())
            self.assertTrue(Path(path + '.tmp').exists())
