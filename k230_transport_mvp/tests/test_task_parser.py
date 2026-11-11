import random
import unittest

from core.task_parser import build_color_to_buffer_slot, parse_task
from core.types import ValidationError


def random_tasks(count=20):
    rng = random.Random(20271031)
    tasks = []
    for _ in range(count):
        colors = rng.sample(range(1, 7), 3)
        groups = [colors, rng.sample([1, 2, 3], 3), rng.sample(colors, 3), rng.sample([1, 2, 3], 3)]
        tasks.append("+".join("".join(str(value) for value in group) for group in groups))
    return tasks


class TaskParserTests(unittest.TestCase):
    def test_official_example(self):
        task = parse_task("156+123+516+231")
        self.assertEqual(task.batch1_colors, (1, 5, 6))
        self.assertEqual(task.batch2_slots, (2, 3, 1))
        self.assertEqual(task.color_to_buffer_slot, {1: 1, 5: 2, 6: 3})

    def test_twenty_random_legal_tasks(self):
        for payload in random_tasks():
            with self.subTest(payload=payload):
                self.assertEqual(parse_task(payload).payload, payload)

    def test_group_lengths_and_plus_count(self):
        for payload in ("156+123+516", "156+123+516+231+123", "15+123+516+231", "1567+123+516+231"):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                parse_task(payload)

    def test_color_ids(self):
        for payload in ("056+123+506+231", "157+123+517+231", "１２３+123+123+231", "1a6+123+516+231"):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                parse_task(payload)

    def test_duplicate_colors(self):
        with self.assertRaises(ValidationError):
            parse_task("115+123+151+231")

    def test_ring_range_and_duplicates(self):
        for payload in ("156+124+516+231", "156+113+516+231", "156+123+516+232"):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                parse_task(payload)

    def test_mismatched_batches(self):
        with self.assertRaises(ValidationError):
            parse_task("156+123+126+231")

    def test_non_qr_and_whitespace(self):
        for payload in ("", None, 123, "hello", "156+123+516+231\n", " 156+123+516+231"):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                parse_task(payload)

    def test_mapping_validation(self):
        self.assertEqual(build_color_to_buffer_slot([1, 5, 6], [3, 1, 2]), {1: 3, 5: 1, 6: 2})
        for colors, slots in (([1, 1, 5], [1, 2, 3]), ([1, 5, 6], [1, 1, 3]), ([True, 5, 6], [1, 2, 3])):
            with self.assertRaises(ValidationError):
                build_color_to_buffer_slot(colors, slots)
