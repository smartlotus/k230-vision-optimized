import unittest
from k230.tuning.schema import defaults
from tools.board_tuning_scene_check import gate, validate_request


class SceneGateTests(unittest.TestCase):
    def test_missing_frames_colors_motion_or_write_context_cannot_pass(self):
        good = dict(frames=100, found=100, stable=98, inside_manual_face_region=100, detector_errors=0)
        self.assertTrue(gate({'4': good}, [4], {}, {}, None))
        for change in [dict(frames=99), dict(found=99), dict(stable=94), dict(inside_manual_face_region=99), dict(detector_errors=1)]:
            self.assertFalse(gate({'4': dict(good, **change)}, [4], {}, {}, None))
        self.assertFalse(gate({'4': good}, [1, 4], {}, {}, None))
        self.assertFalse(gate({'4': good}, [4], {1: 'capture'}, {}, None))
        self.assertFalse(gate({'4': good}, [4], {}, {1: 'exposure'}, None))
        self.assertFalse(gate({'4': good}, [4], {}, {}, 'TEST_CONTEXT_CHANGED'))

    def test_scene_request_rejects_wrong_source_unmeasured_or_malformed_identity(self):
        p = defaults()
        p['contexts']['TOP_BUFFER']['thresholds']['4'] = [41, 57, -51, -30, 17, 38]
        p['contexts']['TOP_BUFFER']['thresholds']['6'] = [51, 64, -20, 2, -44, -21]
        r = dict(attempt='green-fresh01', scene_id='scene03', profile_file='candidate.json',
                 profile_id=p['profile_id'], colors=[4], centers={'4': [66, 130], '6': [254, 99]}, reference_id=6)
        validate_request(r, p)
        for edit in [dict(profile_file='../candidate.json'), dict(colors=[4, 4]), dict(colors=[True]),
                     dict(profile_id='old'), dict(centers={'4': [66, 130]})]:
            with self.assertRaises(ValueError):
                validate_request(dict(r, **edit), p)
        p['contexts']['TOP_BUFFER']['thresholds']['6'] = None
        with self.assertRaises(ValueError):
            validate_request(r, p)
