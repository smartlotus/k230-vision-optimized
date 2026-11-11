import unittest
from k230.capture_health import CaptureHealth


class CaptureHealthTests(unittest.TestCase):
    def test_repeated_faults_keep_original_error_without_log_flood(self):
        health=CaptureHealth();events=[]
        for i in range(25):
            event=health.failure(1,RuntimeError('snapshot failed(3)' if i==0 else 'timeout'),i*50)
            if event:events.append(event)
        self.assertEqual([r['event'] for r in events],['FIRST_FAILURE','FAILURE_UPDATE'])
        row=health.rows[1]
        self.assertEqual(row['errors'],25);self.assertEqual(row['consecutive_errors'],25)
        self.assertEqual(row['first_error'],'snapshot failed(3)');self.assertEqual(row['last_error'],'timeout')
        self.assertNotIn(2,health.rows)

    def test_recovery_keeps_history_and_next_failure_is_a_new_transition(self):
        health=CaptureHealth();health.failure(1,ValueError('first'),0)
        event=health.success(1,50)
        self.assertEqual(event['event'],'RECOVERED');self.assertEqual(event['errors'],1)
        self.assertEqual(event['consecutive_errors'],0);self.assertEqual(event['first_error'],'first')
        self.assertIsNone(health.success(1,100))
        event=health.failure(1,RuntimeError('second'),150)
        self.assertEqual(event['event'],'FIRST_FAILURE');self.assertEqual(event['errors'],2)
