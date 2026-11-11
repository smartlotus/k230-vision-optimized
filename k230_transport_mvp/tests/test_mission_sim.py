import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from core.mission_state import MissionEngine
from core.task_parser import parse_task
from core.task_planner import build_execution_steps
from core.types import Step
from sim.camera_sim import CameraSim
from sim.field_sim import FieldError, FieldSim
from sim.motion_sim import MotionSim
from sim.run_mission import FAULTS, main, run_mission


def legal_payloads():
    import random
    rng = random.Random(101)
    for _ in range(20):
        colors = rng.sample(range(1, 7), 3)
        yield "+".join("".join(map(str, group)) for group in
                       (colors, rng.sample([1, 2, 3], 3), rng.sample(colors, 3), rng.sample([1, 2, 3], 3)))


class MissionSimTests(unittest.TestCase):
    payload = "156+123+516+231"

    def field(self, capacity=3):
        return FieldSim(parse_task(self.payload), capacity)

    def first_process(self, field):
        field.drive("RAW")
        for color in (1, 5, 6):
            field.pick(1, color, "RAW", None)
        field.drive("PROCESS")
        for color, slot in ((1, 1), (5, 2), (6, 3)):
            field.place(1, color, "PROCESS", slot, 1)

    def test_official_end_to_end_and_display(self):
        report = run_mission()
        self.assertEqual(report["summary"]["status"], "DONE")
        self.assertEqual(report["summary"]["counters"], {"picks": 12, "placements": 12})
        self.assertTrue(report["summary"]["field_complete"])
        self.assertEqual(report["display"]["task"], self.payload)
        self.assertEqual(report["display"]["counters"], {"picks": 12, "placements": 12})

    def test_twenty_random_tasks_all_capacities_and_starts(self):
        for payload in legal_payloads():
            for capacity in (1, 2, 3):
                for start in ("START_1", "START_2"):
                    with self.subTest(task=payload, capacity=capacity, start=start):
                        summary = run_mission(payload, start, capacity)["summary"]
                        self.assertEqual(summary["status"], "DONE")
                        self.assertTrue(summary["field_complete"])
                        self.assertEqual(summary["station"], start)
                        self.assertLessEqual(summary["max_carrier_occupancy"], capacity)

    def test_all_fault_injections_fail_without_false_success(self):
        for fault in FAULTS:
            with self.subTest(fault=fault):
                report = run_mission(inject=fault)
                self.assertEqual(report["summary"]["status"], "ERROR")
                self.assertFalse(report["summary"]["field_complete"])
                self.assertIsNone(report["display"]["counters"])
                self.assertFalse(any(e["event"] == "MISSION_DONE" for e in report["events"]))

    def test_ack_loss_preserves_uncertainty_without_retry_or_count(self):
        report = run_mission(inject="ack-loss")
        self.assertEqual(report["summary"]["counters"], {"picks": 0, "placements": 0})
        self.assertEqual(len(report["transitions"]), 1)
        self.assertEqual(len(report["field"]["carrier"]), 1)
        self.assertEqual(report["receipts"][-1]["status"], "UNKNOWN")
        self.assertIsNotNone(report["summary"]["pending_command"])

    def test_timeout_does_not_mutate_or_count(self):
        report = run_mission(inject="timeout")
        self.assertEqual(report["transitions"], [])
        self.assertEqual(report["summary"]["counters"]["picks"], 0)

    def test_process_barrier_even_with_capacity_one(self):
        field = self.field(1)
        field.drive("RAW")
        field.pick(1, 1, "RAW", None)
        field.drive("PROCESS")
        field.place(1, 1, "PROCESS", 1, 1)
        with self.assertRaisesRegex(FieldError, "PROCESS_PHASE_INCOMPLETE"):
            field.pick(1, 1, "PROCESS", 1)

    def test_second_batch_barrier(self):
        field = self.field()
        field.drive("RAW")
        with self.assertRaisesRegex(FieldError, "FIRST_BATCH_INCOMPLETE"):
            field.pick(2, 5, "RAW", None)

    def test_carrier_full(self):
        field = self.field(1)
        field.drive("RAW")
        field.pick(1, 1, "RAW", None)
        before = field.snapshot()
        with self.assertRaisesRegex(FieldError, "CARRIER_FULL"):
            field.pick(1, 5, "RAW", None)
        self.assertEqual(before, field.snapshot())

    def test_wrong_color_and_pick_order(self):
        field = self.field()
        field.drive("RAW")
        with self.assertRaisesRegex(FieldError, "PICK_ORDER_MISMATCH"):
            field.pick(1, 5, "RAW", None)

    def test_wrong_source_station(self):
        with self.assertRaisesRegex(FieldError, "WRONG_SOURCE_STATION"):
            self.field().pick(1, 1, "RAW", None)

    def test_place_requires_cargo(self):
        field = self.field()
        field.drive("PROCESS")
        with self.assertRaisesRegex(FieldError, "CARGO_MISSING"):
            field.place(1, 1, "PROCESS", 1, 1)

    def test_gripper_cannot_cross_stations(self):
        field = self.field()
        field.gripper = "TEST_HELD"
        with self.assertRaisesRegex(FieldError, "GRIPPER_HELD_DURING_TRAVEL"):
            field.drive("PROCESS")

    def test_missing_first_layer_refuses_stack(self):
        field = self.field()
        self.first_process(field)
        for color, slot in ((1, 1), (5, 2), (6, 3)):
            field.pick(1, color, "PROCESS", slot)
        field.drive("BUFFER")
        for color, slot in ((1, 1), (5, 2), (6, 3)):
            field.place(1, color, "BUFFER", slot, 1)
        field.drive("RAW")
        for color in (5, 1, 6):
            field.pick(2, color, "RAW", None)
        field.drive("PROCESS")
        for color, slot in ((5, 2), (1, 3), (6, 1)):
            field.place(2, color, "PROCESS", slot, 1)
        for color, slot in ((5, 2), (1, 3), (6, 1)):
            field.pick(2, color, "PROCESS", slot)
        field.drive("BUFFER")
        field.buffer[2] = {}
        with self.assertRaisesRegex(FieldError, "MATCHING_FIRST_LAYER_REQUIRED"):
            field.place(2, 5, "BUFFER", 2, 2)

    def test_wrong_ring_refuses_without_mutation(self):
        field = self.field()
        field.drive("RAW")
        field.pick(1, 1, "RAW", None)
        field.drive("PROCESS")
        before = field.snapshot()
        with self.assertRaisesRegex(FieldError, "WRONG_RING_OR_LAYER"):
            field.place(1, 1, "PROCESS", 3, 1)
        self.assertEqual(field.snapshot(), before)

    def test_material_transitions_have_no_teleport(self):
        transitions = run_mission()["transitions"]
        self.assertEqual(len(transitions), 24)
        for batch in (1, 2):
            for color in (1, 5, 6):
                route = [(e["source"], e["dest"]) for e in transitions if e["object"] == "B%d:C%d" % (batch, color)]
                self.assertEqual(route, [("RAW", "CARRIER"), ("CARRIER", "PROCESS"), ("PROCESS", "CARRIER"), ("CARRIER", "BUFFER")])

    def test_qr_confirmation_precedes_pick(self):
        events = run_mission()["events"]
        confirmed = next(i for i, e in enumerate(events) if e["event"] == "TASK_CONFIRMED")
        self.assertEqual(sum(e["event"] == "OBSERVATION" for e in events[:confirmed]), 3)
        first_pick = next(i for i, e in enumerate(events) if e["event"] == "COMMAND" and e["step"]["action"] == "PICK")
        self.assertGreater(first_pick, confirmed)

    def test_invalid_qr_cannot_move_to_raw(self):
        field = self.field()
        motion = MotionSim(field)
        engine = MissionEngine(motion, CameraSim("garbage"))
        result = engine.run(self.payload)
        self.assertEqual(result["reason"], "QR_NOT_CONFIRMED")
        self.assertEqual(field.station, "QR")
        self.assertEqual(len(motion.receipts), 1)

    def test_duplicate_receipt_does_not_double_count(self):
        field = self.field()
        engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
        receipt = {"status": "DONE", "command_id": 1}
        self.assertIsNone(engine._confirm(receipt, "PICK"))
        self.assertEqual(engine._confirm(receipt, "PICK"), "STALE_OR_INVALID_RECEIPT")
        self.assertEqual(engine.stats.picks, 1)

    def test_busy_and_error_do_not_count(self):
        for status in ("BUSY", "ERROR", "UNKNOWN", "ACK"):
            field = self.field()
            engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
            self.assertIsNotNone(engine._confirm({"status": status, "command_id": 1}, "PLACE"))
            self.assertEqual(engine.stats.placements, 0)

    def test_engine_catches_reordered_stage_before_motion(self):
        field = self.field()
        engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
        step = Step("PICK", 1, "BUFFER", "PROCESS", "CARRIER", 1, 1, 0)
        result = engine.run(self.payload, steps=[step])
        self.assertEqual(result["reason"], "PROCESS_PHASE_INCOMPLETE")
        self.assertEqual(field.transitions, [])

    def test_cli_output_json_and_failed_exit(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "report.json"
            self.assertEqual(main(["--output", str(path)]), 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["summary"]["status"], "DONE")
            self.assertEqual(main(["--inject", "missing-color"]), 1)
            self.assertEqual(main(["--task", "bad"]), 2)

    def test_no_replay_of_same_engine(self):
        field = self.field()
        engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
        engine.run(self.payload)
        with self.assertRaisesRegex(ValueError, "cannot be replayed"):
            engine.run(self.payload)

    def test_receipt_for_other_operation_cannot_count(self):
        field = self.field()
        engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
        error = engine._confirm({"status": "DONE", "command_id": 1, "command": "DRIVE"}, "PICK")
        self.assertEqual(error, "RECEIPT_COMMAND_MISMATCH")
        self.assertEqual(engine.stats.picks, 0)

    def test_navigation_uses_explicit_simulated_lane_nodes(self):
        routes = run_mission()["field"]["routes"]
        self.assertTrue(routes)
        self.assertTrue(all(route["is_simulated"] for route in routes))
        self.assertTrue(all("SIM_LANE_HUB" in route["path"] for route in routes))

    def test_stale_observation_refuses_first_pick(self):
        class StaleCamera(CameraSim):
            def detect(self, mode, frame, target_id=None, station=None):
                obs = super().detect(mode, frame, target_id, station)
                if mode != "QR":
                    obs["timestamp_ms"] -= 1000
                return obs
        field = self.field()
        engine = MissionEngine(MotionSim(field), StaleCamera(self.payload))
        self.assertEqual(engine.run(self.payload)["reason"], "STALE_OR_MISMATCHED_OBSERVATION")
        self.assertEqual(field.transitions, [])

    def test_real_observation_cannot_enter_synthetic_motion(self):
        class RealCamera(CameraSim):
            def detect(self, mode, frame, target_id=None, station=None):
                obs = super().detect(mode, frame, target_id, station)
                if mode != "QR":
                    obs["is_simulated"] = False
                return obs
        field = self.field()
        engine = MissionEngine(MotionSim(field), RealCamera(self.payload))
        self.assertEqual(engine.run(self.payload)["reason"], "OBSERVATION_TARGET_MISMATCH")
        self.assertEqual(field.transitions, [])

    def test_top_target_is_color_not_buffer_slot(self):
        observations = [event["observation"] for event in run_mission()["events"]
                        if event["event"] == "OBSERVATION" and event["observation"]["kind"] == "TOP"]
        self.assertEqual([obs["target_id"] for obs in observations], [5, 1, 6])
        slots = parse_task(self.payload).color_to_buffer_slot
        self.assertEqual([slots[obs["target_id"]] for obs in observations], [2, 1, 3])

    def test_repeated_qr_frame_cannot_confirm_three_times(self):
        class FrozenCamera(CameraSim):
            def get_frame(self, camera_id):
                return {"camera_id": camera_id, "frame_id": 1, "timestamp_ms": 40, "synthetic": True}
        field = self.field()
        engine = MissionEngine(MotionSim(field), FrozenCamera(self.payload))
        self.assertEqual(engine.run(self.payload)["reason"], "QR_NOT_CONFIRMED")
        self.assertEqual(field.transitions, [])

    def test_unknown_receipt_with_empty_reason_never_commits(self):
        for reason in (None, "", False):
            field = self.field()
            engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
            self.assertEqual(engine._confirm({"status": "UNKNOWN", "command_id": 1, "reason": reason}, "PICK"), "MOTION_UNKNOWN")
            self.assertEqual(engine.stats.picks, 0)

    def test_truthy_non_boolean_source_is_rejected(self):
        class BadSourceCamera(CameraSim):
            def detect(self, mode, frame, target_id=None, station=None):
                obs = super().detect(mode, frame, target_id, station)
                obs["is_simulated"] = "false"
                return obs
        field = self.field()
        engine = MissionEngine(MotionSim(field), BadSourceCamera(self.payload))
        self.assertEqual(engine.run(self.payload)["reason"], "QR_NOT_CONFIRMED")
        self.assertEqual(field.transitions, [])

    def test_explicit_real_receipt_and_missing_receipt_rejected(self):
        field = self.field()
        engine = MissionEngine(MotionSim(field), CameraSim(self.payload))
        self.assertEqual(engine._confirm(None, "PICK"), "INVALID_RECEIPT")
        self.assertEqual(engine._confirm({"status": "DONE", "command_id": 1, "is_simulated": False}, "PICK"), "RECEIPT_SOURCE_MISMATCH")
        self.assertEqual(engine.stats.picks, 0)
