"""Differential tracking and inherited contract checks using host image doubles."""
import copy
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import test_block_detector as native_tests
import test_block_service as service_tests
from k230.block_profile import PARAMETERS, make_config
from k230.vision_adapter import VisionAdapter as LegacyAdapter
from k230.vision_adapter_optimized import VisionAdapter
from vision.block_detector import BlockDetector as Legacy
from vision.block_detector_optimized import BlockDetector


def candidate(x, y=50, color=1, eligible=True):
    return dict(target_id=color, center=(x, y), eligible=eligible,
                bbox=(x-10, y-10, 20, 20), stable=False)


class OptimizedNativeDetectionTests(native_tests.BlockTests):
    def setUp(self):
        replacement = patch.object(native_tests, 'BlockDetector', BlockDetector)
        replacement.start()
        self.addCleanup(replacement.stop)
        super().setUp()


class OptimizedServiceTests(service_tests.BlockServiceTests):
    def setUp(self):
        replacement = patch.object(service_tests, 'VisionAdapter', VisionAdapter)
        replacement.start()
        self.addCleanup(replacement.stop)


class BlockDifferentialTests(unittest.TestCase):
    def assert_update_equal(self, old, new, candidates, frame, timestamp):
        original = copy.deepcopy(candidates)
        previous = copy.deepcopy(new.tracks)
        previous_reference = new.tracks
        self.assertEqual(old.update(candidates, frame, timestamp),
                         new.update(candidates, frame, timestamp))
        self.assertEqual(old.tracks, new.tracks)
        self.assertEqual(old.next_id, new.next_id)
        self.assertEqual(candidates, original)
        self.assertEqual(previous_reference, previous)

    def test_random_sequences_and_runtime_window_changes(self):
        for seed in range(10):
            rng = random.Random(seed)
            old, new = Legacy(copy.deepcopy(PARAMETERS)), BlockDetector(copy.deepcopy(PARAMETERS))
            frame = timestamp = 0
            for i in range(300):
                if i % 37 == 0:
                    frames = rng.choice((1, 2, 3, 10, 20))
                    old.p['stable_frames'] = new.p['stable_frames'] = frames
                frame += rng.choice((0, 1, 1, 1, -1))
                timestamp += rng.choice((0, 20, 40, 1000, -10))
                rows = [candidate(rng.randrange(30, 100) / 2,
                                  rng.randrange(30, 100) / 2,
                                  rng.choice((1, 1, 1, 2, 6)), rng.choice((True, True, False)))
                        for _ in range(rng.randrange(13))]
                self.assert_update_equal(old, new, rows, frame, timestamp)

    def test_ambiguity_boundary_exact_ties_and_track_distance(self):
        for positions in ((0, 0), (0, 4), (0, 5), (-4, 4), (4, 0), (5, 0), (28, 29)):
            old, new = Legacy(copy.deepcopy(PARAMETERS)), BlockDetector(copy.deepcopy(PARAMETERS))
            self.assert_update_equal(old, new, [candidate(x) for x in positions], 1, 40)
            self.assert_update_equal(old, new, [candidate(0), candidate(1)], 2, 80)

    def test_stability_boundary_and_misses(self):
        old, new = Legacy(copy.deepcopy(PARAMETERS)), BlockDetector(copy.deepcopy(PARAMETERS))
        motion = old.p['stable_motion_px']
        for i, rows in enumerate(([candidate(0)], [candidate(motion)],
                                   [candidate(0)], [candidate(motion+.01)], [],
                                   [candidate(0, eligible=False)])):
            self.assert_update_equal(old, new, rows, i+1, (i+1)*40)

    def test_adapter_results_and_optimized_detector_selection(self):
        old, new = LegacyAdapter(make_config()), VisionAdapter(make_config())
        image = native_tests.Image([native_tests.Blob(1, (20, 60, 30, 50), 1100)])
        for i in range(1, 6):
            frame = service_tests.frame(image, i)
            self.assertEqual(old.detect('COLOR', frame, 1, 'RAW'), new.detect('COLOR', frame, 1, 'RAW'))
            self.assertEqual(old.last_block_candidates, new.last_block_candidates)
        self.assertIsInstance(next(iter(new.blocks.values())), BlockDetector)
        new.reset_session()
        self.assertEqual(new.blocks, {})

    def test_new_entries_reject_host_and_cached_installation(self):
        from k230.main_vision_optimized import main
        with self.assertRaises(SystemExit):
            main()
        source = Path(__file__).resolve().parents[1] / 'run_vision_service_optimized.py'
        code = compile(source.read_text(encoding='utf-8'), str(source), 'exec')
        with self.assertRaisesRegex(SystemExit, 'Use CanMV'):
            exec(code, {'__name__': '__main__'})
        with patch.object(sys, 'implementation', types.SimpleNamespace(name='micropython')):
            with self.assertRaisesRegex(SystemExit, 'already loaded'):
                exec(code, {'__name__': '__main__'})
