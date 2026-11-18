"""Regression checks for the native bench's artifact and path boundaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from board_tensor_bench import isolated_root
from tensor_bench import verify


class NativeBenchBoundaries(unittest.TestCase):
    def test_independent_sd_root(self):
        self.assertTrue(isolated_root("/sdcard/xy1117c"))
        for root in ("/sdcard/xy/../transport_mvp", "/sdcard/xy1117c/..", "/sdcard/xy1117c/",
                     "/sdcard/transport_mvp", "/sdcard/xy", "/sdcard/xy../main.py"):
            with self.subTest(root=root):
                self.assertFalse(isolated_root(root))

    def test_foreign_cpu_replay_model_rejected(self):
        evidence = Path(__file__).resolve().parents[1] / "evidence/G7-native-input-20261117"
        spec = json.loads((evidence / "inputs-after/spec.json").read_text(encoding="utf8"))
        foreign = SimpleNamespace(checkpoint_sha="0" * 64, manifest_sha=spec["manifest_sha256"],
                                  cal=spec["primary_cal"], set_cal=spec["set_cal"],
                                  primary_qualified=spec["primary_qualified"])
        args = argparse.Namespace(inputs=evidence / "inputs-after", native=evidence / "final",
                                  provider=Path("unused"), out=Path("must-not-be-written.json"))
        with patch("tensor_bench.localizer", return_value=foreign):
            with self.assertRaisesRegex(ValueError, "CPU replay provider differs"):
                verify(args)


if __name__ == "__main__":
    unittest.main()
