"""Native evidence must fail closed when counts or engineering boundaries drift."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_usb_smoke import sha, validate_runs, verify_report


class NativeEvidenceGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1]
        cls.path = base / "evidence/USB-native-20261117-e/chain30-f01.json"
        cls.deployment = base / "evidence/USB-engineering-20261117-d/engineering.json"
        cls.document = json.loads(cls.path.read_text(encoding="utf8"))
        cls.sources = cls.document["sources"]

    def test_actual_native_report(self):
        result = verify_report(self.path, 30,
            "7a28e000df73f522ba23c420432a58f5249f4c3260827d45eca2e93edbe19128",
            self.deployment, self.sources)
        self.assertEqual(result["frames"], 30)

    def test_sd_card_report_without_startup_files_is_valid(self):
        base = Path(__file__).resolve().parents[1]
        path = base / "evidence/USB-native-20261118-sd/chain30-sd1118.json"
        result = verify_report(path, 30,
            "0dcdbbd52efbfff13230628d0f94f8e72539c879ccb586d13d5015c96cea0728",
            base / "evidence/USB-engineering-20261117-d/engineering.json",
            json.loads(path.read_text(encoding="utf8"))["sources"])
        self.assertEqual(result["frames"], 30)

    def test_incomplete_stage_rejected_despite_complete_status(self):
        document = copy.deepcopy(self.document)
        document["kpu_calls"] = 29
        self.reject(document)

    def test_calibrated_claim_rejected(self):
        document = copy.deepcopy(self.document)
        document["last_output"]["confidence"] = 0.9
        self.reject(document)

    def test_unreleased_pool_rejected(self):
        document = copy.deepcopy(self.document)
        document["final_memory"]["sys_mmz"]["used"] = 4096
        self.reject(document)

    def test_missing_timing_sample_rejected(self):
        document = copy.deepcopy(self.document)
        document["timing"]["cycle"]["samples"] = 29
        self.reject(document)

    def test_stale_source_rejected(self):
        sources = dict(self.sources, **{"board_usb_smoke.py":"0" * 64})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            verify_report(self.path, 30, sha(self.path), self.deployment, sources)

    def test_empty_run_inventory_rejected(self):
        with self.assertRaises(ValueError):
            validate_runs([])

    def test_duplicate_native_evidence_rejected(self):
        item = dict(file="run.json", frames=30, native_sha256="a" * 64)
        with self.assertRaises(ValueError):
            validate_runs([item, dict(item)])
        with self.assertRaises(ValueError):
            validate_runs([item, dict(item, file="other.json")])

    def test_invalid_frame_denominators_rejected(self):
        for count in (0, -1, True, 30.0, 3001):
            with self.subTest(frames=count), self.assertRaises(ValueError):
                validate_runs([dict(file="run.json", frames=count, native_sha256="a" * 64)])

    def reject(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.json"
            path.write_text(json.dumps(document), encoding="utf8")
            with self.assertRaises(ValueError):
                verify_report(path, 30, sha(path), self.deployment, self.sources)


if __name__ == "__main__":
    unittest.main()
