import copy
import json
from pathlib import Path
import unittest
from tools.validate_reconnect_report import validate

DIRECTORY = Path(__file__).resolve().parents[1]/'evidence/board'


def reports():
    return [json.loads((DIRECTORY/name).read_text(encoding='utf-8')) for name in
            ('lifecycle-reconnect.json', 'native-fixtures-reconnect.json', 'deployment-reconnect.json')]


class ReconnectEvidenceTests(unittest.TestCase):
    def test_readbacks_prove_only_the_measured_scopes(self):
        report = validate(*reports())
        self.assertEqual(report['status'], 'PASS')
        self.assertFalse(report['physical_uart_validated'])
        self.assertFalse(report['live_target_accuracy_validated'])
        self.assertFalse(report['live_mm_calibration_validated'])

    def test_replayed_or_bad_control_acceptance_cannot_pass(self):
        for change in ('not_rejected', 'wrong_mode', 'wrong_ack'):
            lifecycle, fixture, deployment = reports()
            control = lifecycle['phases'][0]
            if change == 'not_rejected':
                control['rejected_control_frames'] = 0
            elif change == 'wrong_mode':
                control['observations'][1]['kind'] = 'TOP'
            else:
                control['acks'].append({'type': 'A', 'command_seq': 60000, 'status': 'ACK'})
            with self.assertRaises(ValueError):
                validate(lifecycle, fixture, deployment)

    def test_failure_or_unapproved_mm_never_gets_success_summary(self):
        for change in ('incomplete', 'errors', 'millimeters'):
            lifecycle, fixture, deployment = reports()
            if change == 'incomplete':
                lifecycle['finished'] = False
            elif change == 'errors':
                lifecycle['errors'] = ['camera cleanup failure']
            else:
                lifecycle['phases'][0]['observations'][0]['x_mm'] = 1.0
            with self.assertRaises(ValueError):
                validate(lifecycle, fixture, deployment)

    def test_source_or_scope_mismatch_is_rejected(self):
        for change in ('hash', 'fixture_source', 'physical_uart'):
            lifecycle, fixture, deployment = reports()
            if change == 'hash':
                deployment['actual']['k230/main.py'] = 'incorrect'
            elif change == 'fixture_source':
                fixture['is_simulated'] = False
            else:
                lifecycle['physical_uart_enabled'] = True
            with self.assertRaises(ValueError):
                validate(lifecycle, fixture, deployment)


if __name__ == '__main__':
    unittest.main()
