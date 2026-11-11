import unittest
from run_multicam_preview import render_round, new_states


class Frame:
    def width(self): return 640
    def height(self): return 480


class Sensor:
    def __init__(self, values):
        self.values = list(values)
        self.timeouts = []
    def snapshot(self, timeout, auto_reboot):
        self.timeouts.append((timeout, auto_reboot))
        value = self.values.pop(0)
        if isinstance(value, BaseException): raise value
        return value


class Canvas:
    def __init__(self):
        self.images = []
        self.labels = []
    def draw_image(self, frame, x, y, **kwargs): self.images.append((x,y))
    def draw_string_advanced(self, x, y, size, text, **kwargs): self.labels.append(text)


class PreviewRecoveryTests(unittest.TestCase):
    def test_flagged_camera_is_never_sampled_even_if_supplied(self):
        from run_multicam_preview import CAMERA_IDS, QUARANTINED_CAMERA_IDS
        self.assertEqual(CAMERA_IDS, (2,1,0))
        self.assertEqual(QUARANTINED_CAMERA_IDS, (0,))
        sensors={2:Sensor([Frame()]),1:Sensor([Frame()]),0:Sensor([Frame()])}
        states,canvas=new_states(),Canvas()
        render_round(sensors,canvas,states,quarantined_ids=QUARANTINED_CAMERA_IDS)
        self.assertEqual(sensors[0].timeouts,[])
        self.assertEqual(states[0]['status'],'QUARANTINED')
        self.assertIn('CSI0 FLAGGED',canvas.labels)
        self.assertEqual([states[c]['frames'] for c in (2,1)],[1,1])

    def test_one_camera_failure_keeps_other_two_live(self):
        sensors={2:Sensor([Frame()]),0:Sensor([RuntimeError('snapshot failed(3)')]),1:Sensor([Frame()])}
        states, canvas=new_states(),Canvas()
        render_round(sensors,canvas,states)
        self.assertEqual([states[c]['frames'] for c in (2,0,1)],[1,0,1])
        self.assertEqual(states[0]['errors'],1)
        self.assertIn('CSI0 RETRY',canvas.labels)
        self.assertEqual(canvas.images,[(40,0),(40,240)])

    def test_failed_camera_recovers_without_resetting_streams(self):
        sensors={2:Sensor([Frame(),Frame()]),0:Sensor([RuntimeError('timeout'),Frame()]),1:Sensor([Frame(),Frame()])}
        states=new_states()
        render_round(sensors,Canvas(),states)
        render_round(sensors,Canvas(),states)
        self.assertEqual(states[0]['frames'],1)
        self.assertEqual(states[0]['errors'],1)
        self.assertEqual(states[0]['consecutive_errors'],0)
        self.assertEqual(sensors[0].timeouts,[(2000,False),(500,False)])
        self.assertEqual(states[1]['frames'],2)

    def test_user_stop_is_not_swallowed_as_camera_failure(self):
        for error in (KeyboardInterrupt(),RuntimeError('IDE interrupt')):
            sensors={2:Sensor([error]),0:Sensor([Frame()]),1:Sensor([Frame()])}
            with self.assertRaises(type(error)):
                render_round(sensors,Canvas(),new_states())
            self.assertEqual(sensors[0].timeouts,[])

    def test_no_frame_is_not_drawn_as_live_image(self):
        sensors={2:Sensor([None]),0:Sensor([Frame()]),1:Sensor([Frame()])}
        states,canvas=new_states(),Canvas()
        render_round(sensors,canvas,states)
        self.assertEqual(states[2]['errors'],1)
        self.assertEqual(states[2]['frames'],0)
        self.assertNotIn((40,0),canvas.images)

    def test_missing_camera_is_offline_without_blocking_other_views(self):
        sensors={2:Sensor([Frame()]),1:Sensor([Frame()])}
        states,canvas=new_states(),Canvas()
        render_round(sensors,canvas,states)
        self.assertEqual(states[0]['status'],'OFFLINE')
        self.assertEqual(states[0]['attempts'],0)
        self.assertIn('CSI0 OFFLINE',canvas.labels)
        self.assertEqual(states[2]['status'],'LIVE')
        self.assertEqual(states[1]['status'],'LIVE')


if __name__=='__main__': unittest.main()
