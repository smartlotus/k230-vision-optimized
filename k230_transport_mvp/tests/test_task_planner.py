import unittest

from core.task_parser import parse_task
from core.task_planner import build_execution_steps
from core.types import ValidationError


class TaskPlannerTests(unittest.TestCase):
    def setUp(self):
        self.task = parse_task("156+123+516+231")

    def test_first_batch_matching_rings(self):
        steps = build_execution_steps(self.task)
        for dest in ("PROCESS", "BUFFER"):
            self.assertEqual([s.target_slot for s in steps if s.action == "PLACE" and s.batch == 1 and s.dest_station == dest], [1, 2, 3])

    def test_second_batch_process_uses_fourth_group(self):
        self.assertEqual([s.target_slot for s in build_execution_steps(self.task)
                          if s.action == "PLACE" and s.batch == 2 and s.dest_station == "PROCESS"], [2, 3, 1])

    def test_second_batch_stack_uses_color_mapping(self):
        steps = [s for s in build_execution_steps(self.task) if s.action == "PLACE" and s.batch == 2 and s.dest_station == "BUFFER"]
        self.assertEqual([s.target_slot for s in steps], [2, 1, 3])
        self.assertTrue(all(s.layer == 2 for s in steps))

    def test_every_capacity_respects_phase_barriers(self):
        for capacity in (1, 2, 3):
            steps = build_execution_steps(self.task, capacity)
            for batch in (1, 2):
                placements = [i for i, s in enumerate(steps) if s.batch == batch and s.action == "PLACE" and s.dest_station == "PROCESS"]
                retrievals = [i for i, s in enumerate(steps) if s.batch == batch and s.action == "PICK" and s.source_station == "PROCESS"]
                self.assertLess(max(placements), min(retrievals))
            first_buffer = [i for i, s in enumerate(steps) if s.batch == 1 and s.action == "PLACE" and s.dest_station == "BUFFER"]
            second_raw = [i for i, s in enumerate(steps) if s.batch == 2 and s.action == "PICK" and s.source_station == "RAW"]
            self.assertLess(max(first_buffer), min(second_raw))

    def test_capacity_and_load_state(self):
        for capacity in (1, 2, 3):
            occupancy = 0
            for step in build_execution_steps(self.task, capacity):
                occupancy += int(step.action == "PICK") - int(step.action == "PLACE")
                self.assertLessEqual(occupancy, capacity)
                self.assertGreaterEqual(occupancy, 0)
                if step.action == "MOVE" and "CARRIER_EMPTY" in step.preconditions:
                    self.assertEqual(occupancy, 0)
            self.assertEqual(occupancy, 0)

    def test_invalid_capacity(self):
        for capacity in (0, 4, -1, 1.0, True, False, "3"):
            with self.subTest(capacity=capacity), self.assertRaises(ValidationError):
                build_execution_steps(self.task, capacity)

    def test_complete_action_denominators(self):
        steps = build_execution_steps(self.task)
        self.assertEqual(sum(s.action == "PICK" for s in steps), 12)
        self.assertEqual(sum(s.action == "PLACE" for s in steps), 12)
        self.assertEqual(steps[-1].action, "RETURN")
