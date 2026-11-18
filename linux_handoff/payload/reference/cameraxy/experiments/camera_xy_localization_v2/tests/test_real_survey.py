"""Regression coverage for independent survey intake, reporting and export."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from contract import digest, load_rows
from export_k230 import check_acceptance
from infer import real_evaluate
from prepare_real import create_manifest


class SurveyReporting(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.survey_path = self.root / "survey.json"
        self.manifest = self.root / "frames.jsonl"
        Image.new("RGB", (16, 9), "#ff0000").save(self.root / "a.png")
        self.survey = {
            "camera_metadata": {"sensor": "USB_OV2735", "capture_source": "SYNTHETIC_FIXTURE"},
            "positions": [dict(survey_position_id="fixture-1", camera_xy_target={
                "camera_x_mm": 0, "camera_y_mm": 0},
                measurement=dict(reference="DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY",
                                 method="fixture only", record_id="fixture", uncertainty_mm=2),
                images=[dict(sample_id="frame-1", image="a.png")])],
        }
        cal = dict(bin_edges=[], probabilities=[.75], threshold=.3)
        self.model = SimpleNamespace(
            cal=cal, set_cal=cal, primary_qualified=True, manifest_sha="train", checkpoint_sha="model",
            predict_survey=Mock(return_value=(
                np.array([[60, 0], [40, 0], [200, 0], [400, 0], [500, 0]]), .9, .95)))

    def prepare(self):
        self.survey_path.write_text(json.dumps(self.survey), encoding="utf-8")
        return create_manifest(self.survey_path, self.manifest)

    def test_fixture_has_set_metrics_and_cannot_accept_primary(self):
        self.prepare()
        report = real_evaluate(self.model, self.manifest)
        self.assertEqual(report["status"], "SYNTHETIC_SURVEY_FIXTURE")
        self.assertEqual(report["contents_sha256"], digest(self.manifest))
        self.assertEqual(report["metrics"]["accepted_count"], 0)
        self.assertEqual(report["candidate_set_metrics"]["success_within50"], 1)
        self.assertEqual(report["candidate_set_metrics"]["brier50"], .0625)
        self.assertEqual(report["observations"][0]["candidate_error_mm"], [60, 40, 200, 400, 500])
        self.assertEqual(report["observations"][0]["candidate_set_confidence"], .75)

    def test_legacy_manifest_uses_the_frozen_primary_qualification(self):
        del self.survey["camera_metadata"]["capture_source"]
        self.model.primary_qualified = False
        self.prepare()
        report = real_evaluate(self.model, self.manifest)
        self.assertEqual(report["capture_source"], "OPERATOR_OV2735")
        self.assertEqual(report["metrics"]["coverage"], 0)
        self.assertFalse(report["observations"][0]["primary_accepted"])

    def test_direct_path_has_no_invented_set_confidence(self):
        self.prepare()
        self.model.set_cal = None
        self.model.predict_survey.return_value = np.array([[0, 0]]), .8, None
        report = real_evaluate(self.model, self.manifest)
        self.assertIsNone(report["candidate_set_metrics"])
        self.assertIsNone(report["observations"][0]["candidate_set_confidence"])

    def test_duplicate_bytes_rejected_by_both_entrances(self):
        shutil.copyfile(self.root / "a.png", self.root / "b.png")
        images = self.survey["positions"][0]["images"]
        images.append(dict(sample_id="frame-2", image="b.png"))
        with self.assertRaisesRegex(ValueError, "Duplicate survey image bytes"):
            self.prepare()
        self.assertFalse(self.manifest.exists())
        images.pop()
        self.prepare()
        row = load_rows(self.manifest)[0]
        other = dict(row, sample_id="frame-2", image="b.png")
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(other) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate survey image bytes"):
            real_evaluate(self.model, self.manifest)
        self.model.predict_survey.assert_not_called()

    def test_robot_label_never_becomes_camera_ground_truth(self):
        self.survey["positions"][0]["measurement"]["reference"] = "ROBOT_CHASSIS_CENTER"
        with self.assertRaisesRegex(ValueError, "measured camera optical center"):
            self.prepare()
        self.assertFalse(self.manifest.exists())

    def test_export_requires_actual_accepted_research_metrics(self):
        self.manifest.write_text("{}\n", encoding="utf-8")
        manifest = self.root / "manifest.jsonl"
        self.manifest.rename(manifest)
        identity = digest(manifest)
        gate = self.root / "host.json"
        real_report = self.root / "real.json"
        gate.write_text(json.dumps(dict(
            status="COMPUTE_COMPLETE", full_gate="PASS", model_selected_by_validation_median="direct",
            dataset_manifest_sha256=identity,
            scientific_goal=dict(accepted_coverage_min=.25, median_mm=20, p95_mm=50))), encoding="utf-8")
        source = dict(status="MEASURED_REAL_OV2735", capture_source="OPERATOR_OV2735",
                      primary_qualified=True, checkpoint_sha256="model", training_manifest_sha256=identity,
                      metrics=dict(accepted_count=5, coverage=.5, accepted_median_mm=15, accepted_p95_mm=25))
        args = SimpleNamespace(data=self.root, host_acceptance=gate, real_report=real_report)
        localizer = SimpleNamespace(cell_mm=0, checkpoint_sha="model", manifest_sha=identity)
        real_report.write_text(json.dumps(source), encoding="utf-8")
        check_acceptance(args, localizer)
        for change in (dict(status="SYNTHETIC_SURVEY_FIXTURE"), dict(primary_qualified=False),
                       dict(metrics=dict(source["metrics"], coverage=.24)),
                       dict(metrics=dict(source["metrics"], accepted_p95_mm=51))):
            with self.subTest(change=change):
                real_report.write_text(json.dumps(dict(source, **change)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "host/real validation"):
                    check_acceptance(args, localizer)


if __name__ == "__main__":
    unittest.main()
