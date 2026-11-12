import unittest
from unittest.mock import patch
from tools.frozen_model_fps import metrics,run
from tools.prepare_frozen_model_fps import upload_bundle


class FrozenModelTests(unittest.TestCase):
    def test_measured_latency_and_inverse_rate(self):
        result=metrics([10,20,30,40])
        self.assertEqual(result['mean_ms'],25)
        self.assertEqual(result['fps'],40)
        self.assertEqual(result['p95_ms'],40)
        with self.assertRaises(ValueError):metrics([])
        with self.assertRaises(ValueError):metrics([-1])

    def test_invalid_iteration_budget_prevents_native_import(self):
        with self.assertRaises(ValueError):run('/absent',iterations=100)

    def test_upload_refuses_original_directory(self):
        with self.assertRaises(ValueError):upload_bundle('/absent','/sdcard/transport_mvp','/absent/receipt')
