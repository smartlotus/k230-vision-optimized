import unittest
from k230.main import run_loop


class Camera:
    close_errors=[]
    def __init__(self,fail=False):
        self.closed=False
        self.fail=fail
        self.captures=0
    def start(self):
        if self.fail:
            raise RuntimeError('start failed')
    def camera_for_mode(self,mode):
        return 2
    def get_frame(self,camera_id):
        self.captures+=1
        return {'camera_id':camera_id,'frame_id':self.captures}
    def close(self):
        self.closed=True


class Vision:
    def __init__(self):
        self.calls=[]
    def detect(self,mode,frame,target_id=None,station=None):
        self.calls.append((mode,target_id,station))
        return {'status':'UNCALIBRATED','kind':mode,'x_mm':None,'y_mm':None}


class Uart:
    enabled=True
    def __init__(self,commands):
        self.commands=list(commands)
        self.closed=False
        self.observations=[]
        self.acks=[]
        self.beats=0
    def poll(self):
        return self.commands.pop(0) if self.commands else []
    def heartbeat(self,now):
        self.beats+=1
    def send_ack(self,*args):
        self.acks.append(args)
    def send_observation(self,observation):
        self.observations.append(observation)
        return True
    def close(self):
        self.closed=True


class RuntimeTests(unittest.TestCase):
    def run_runtime(self,camera,vision,uart,limit=5):
        return run_loop(camera,vision,uart,{},clock_ms=lambda:100,
                        sleep_ms=lambda n:None,emit=lambda x:None,max_iterations=limit)

    def test_stop_processed_before_next_capture_and_resources_closed(self):
        camera,vision=Camera(),Vision()
        uart=Uart([[],[{'seq':1,'command':'STOP'}]])
        report=self.run_runtime(camera,vision,uart)
        self.assertEqual(camera.captures,1)
        self.assertEqual(report['reason'],'CONTROL_STOP')
        self.assertTrue(camera.closed and uart.closed)
        self.assertFalse(report['motors_enabled'])

    def test_mode_command_selects_target_and_ping_does_not_move(self):
        camera,vision=Camera(),Vision()
        uart=Uart([[{'seq':2,'command':'MODE','mode':'COLOR','target_id':6,'station':'RAW'},
                    {'seq':3,'command':'PING'}]])
        self.run_runtime(camera,vision,uart,1)
        self.assertEqual(vision.calls,[('COLOR',6,'RAW')])
        self.assertEqual([a[1] for a in uart.acks],['ACK','PONG'])
        self.assertFalse(uart.observations[0]['target_pose_valid'])

    def test_idle_still_polls_heartbeat_and_stop(self):
        uart=Uart([[{'seq':1,'command':'MODE','mode':'IDLE'}],[],[{'seq':2,'command':'STOP'}]])
        camera,vision=Camera(),Vision()
        self.run_runtime(camera,vision,uart)
        self.assertEqual(camera.captures,0)
        self.assertEqual(uart.beats,2)

    def test_initialization_failure_cleans_both_resources(self):
        camera,uart=Camera(True),Uart([])
        with self.assertRaises(RuntimeError):
            self.run_runtime(camera,Vision(),uart)
        self.assertTrue(camera.closed and uart.closed)

    def test_detector_failure_continues_to_process_stop(self):
        class Broken(Vision):
            def detect(self,*args,**kwargs):
                raise RuntimeError('native capture failure')
        uart=Uart([[],[{'seq':1,'command':'STOP'}]])
        self.run_runtime(Camera(),Broken(),uart)
        self.assertEqual(uart.observations[0]['status'],'ERROR')
        self.assertIsNone(uart.observations[0]['x_mm'])


if __name__=='__main__':
    unittest.main()
