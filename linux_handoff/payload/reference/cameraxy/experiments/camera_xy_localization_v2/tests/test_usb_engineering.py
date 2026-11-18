"""Engineering-only model and path contracts must reject production-like artifacts."""
import copy
import json
from pathlib import Path
import unittest

from board_usb_smoke import engineering_contract, statistics, valid_root


class EngineeringBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "evidence/USB-engineering-20261117-d/engineering.json"
        cls.document = json.loads(path.read_text(encoding="utf8"))

    def test_actual_engineering_artifact(self):
        self.assertEqual(len(engineering_contract(self.document)), 932)

    def test_production_and_calibrated_claims_rejected(self):
        for key, value in (("production_eligible", True), ("require_unlocalized", False),
                           ("method", "fused_map"), ("accuracy_test", "PASS")):
            item = copy.deepcopy(self.document)
            item[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    engineering_contract(item)

    def test_nonfinite_anchor_rejected(self):
        item = copy.deepcopy(self.document)
        item["identity"]["map_centers_mm"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            engineering_contract(item)

    def test_root_containment(self):
        self.assertTrue(valid_root("/tmp/xyusb1117d"))
        self.assertTrue(valid_root("/sdcard/xyusb1117d"))
        for path in ("/tmp/xy/../transport_mvp", "/tmp/xyusb1117d/..", "/tmp", "/sdcard/main.py"):
            self.assertFalse(valid_root(path))

    def test_timing_denominator_and_percentile(self):
        result = statistics([1000, 3000, 2000, 4000])
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["p95_ms"], 4)


if __name__ == "__main__":
    unittest.main()
