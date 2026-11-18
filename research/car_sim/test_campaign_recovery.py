"""Fault injection for T00's SQLite/publication boundary."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from campaign_ledger import Ledger, digest, supervisor_lock, write_json


def spec(key, depends=()):
    return {"key": key, "worker": key, "depends": list(depends), "artifacts": ["result.json", "metrics.json"]}


class CampaignRecoveryTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory(prefix="k230-recovery-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.specs = {s["key"]: s for s in (spec("first"), spec("second"), spec("delivery"))}
        self.ledger = Ledger(self.root)
        self.addCleanup(lambda: self.ledger.db.close())
        self.ledger.register(self.specs.values(), "test-frozen-input")

    def publish(self, key):
        attempt, stage = self.ledger.stage(key, 60)
        write_json(stage / "result.json", {"status": "PASS"})
        write_json(stage / "metrics.json", {"measurements": [1, 2, 3]})
        self.ledger.publish(self.specs[key], attempt, stage)
        return self.root / "episodes" / key

    def test_killed_worker_partial_data_resumes_missing_case_only(self):
        previous = self.publish("first")
        stable = digest(previous / "receipt.json")
        attempt, stage = self.ledger.stage("second", 60)
        output = stage / "metrics.json"
        script = ("from pathlib import Path; import sys,time; "
                  "Path(sys.argv[1]).write_text('{'); time.sleep(30)")
        child = subprocess.Popen([sys.executable, "-B", "-c", script, str(output)])
        try:
            wait_until = time.monotonic() + 5
            while not output.exists() and child.poll() is None and time.monotonic() < wait_until:
                time.sleep(.02)
            self.assertIsNone(child.poll())
            self.assertEqual(output.read_text(), "{")
            self.ledger.heartbeat(attempt, child.pid, 60)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        self.ledger.db.close()
        self.ledger = Ledger(self.root)
        self.ledger.recover(self.specs)
        self.assertEqual(digest(previous / "receipt.json"), stable)
        self.assertEqual(len(self.ledger.attempts("first")), 1)
        self.assertEqual(self.ledger.attempts("second")[0]["status"], "INTERRUPTED")
        self.assertFalse((self.root / "episodes" / "second").exists())
        self.assertTrue(output.exists())
        self.publish("second")
        self.assertEqual([a["status"] for a in self.ledger.attempts("second")], ["INTERRUPTED", "PASS"])

    def test_crash_after_atomic_publish_adopts_valid_receipt(self):
        attempt, stage = self.ledger.stage("first", 60)
        write_json(stage / "result.json", {"status": "PASS"})
        write_json(stage / "metrics.json", {"measurements": [1]})
        with patch.object(self.ledger, "mark", side_effect=RuntimeError("crash before database commit")):
            with self.assertRaises(RuntimeError):
                self.ledger.publish(self.specs["first"], attempt, stage)
        self.assertEqual(self.ledger.job("first")["status"], "RUNNING")
        self.ledger.recover(self.specs)
        self.assertEqual(self.ledger.job("first")["status"], "PASS")
        self.assertEqual([a["status"] for a in self.ledger.attempts("first")], ["PASS"])

    def test_corrupted_publication_regenerates_result_and_report(self):
        first = self.publish("first")
        self.publish("second")
        self.publish("delivery")
        (first / "metrics.json").write_text("{}", encoding="utf-8")
        self.ledger.recover(self.specs)
        self.assertEqual(self.ledger.job("first")["status"], "PENDING")
        self.assertEqual(self.ledger.job("second")["status"], "PASS")
        self.assertEqual(self.ledger.job("delivery")["status"], "PENDING")
        self.assertFalse(first.exists())
        self.assertFalse((self.root / "episodes" / "delivery").exists())
        self.assertEqual(len(list((self.root / "attempts" / "first").glob("invalid-*"))), 1)
        self.publish("first")
        self.publish("delivery")
        self.assertEqual([a["status"] for a in self.ledger.attempts("first")], ["INVALID", "PASS"])

    def test_another_supervisor_cannot_claim_live_campaign(self):
        with supervisor_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                with supervisor_lock(self.root):
                    self.fail("A second supervisor acquired the campaign")


if __name__ == "__main__":
    unittest.main()
