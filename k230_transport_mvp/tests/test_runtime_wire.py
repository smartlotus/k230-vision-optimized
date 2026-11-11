"""Integration through the real vision tracker, UART codec and event loop."""
import unittest
from k230.main import run_loop
from k230.vision_adapter import VisionAdapter
from k230.uart_adapter import UartAdapter
from ports.protocol import StreamParser, encode_control


class Code:
    def payload(self):
        return '156+123+516+231'


class Image:
    def width(self):
        return 640
    def height(self):
        return 480
    def to_grayscale(self):
        return self
    def find_qrcodes(self,**kwargs):
        return [Code()]


class Camera:
    close_errors=[]
    def __init__(self,clock):
        self.clock=clock
        self.frame=0
        self.closed=False
    def start(self):
        return self
    def camera_for_mode(self,mode):
        return 2
    def get_frame(self,camera_id):
        self.frame+=1
        return {'image':Image(),'camera_id':2,'frame_id':self.frame,
                'timestamp_ms':self.clock(),'is_simulated':False,
                'image_profile':{'width':640,'height':480,'crop':False,
                                 'hmirror':False,'vflip':False,'rotation':0,'pose_id':None}}
    def close(self):
        self.closed=True


class Wire:
    def __init__(self):
        self.output=bytearray()
    def any(self):
        return 0
    def write(self,data):
        self.output.extend(data)
        return len(data)
    def deinit(self):
        self.closed=True


class RuntimeWireTests(unittest.TestCase):
    def test_qr_three_real_adapter_frames_transmit_complete_observation(self):
        now=[100]
        def clock():
            return now[0]
        def sleep(ms):
            now[0]+=40
        wire=Wire()
        uart=UartAdapter({'enabled':True,'pins_verified':True,'uart_id':2,
                          'tx_pin':11,'rx_pin':12,'max_write_bytes':768},
                          transport=wire,clock_ms=clock)
        camera=Camera(clock)
        vision=VisionAdapter({'CAMERA_IDS':(2,), 'MODE_CAMERA':{'QR':2}})
        report=run_loop(camera,vision,uart,{},clock_ms=clock,sleep_ms=sleep,
                        emit=lambda x:None,max_iterations=4)
        frames=StreamParser().feed(bytes(wire.output),now[0])
        observations=[f for f in frames if f['type']=='V']
        self.assertEqual(len(observations),4)
        submitted=[o for o in observations if o['status']=='OK']
        self.assertEqual(len(submitted),1)
        self.assertEqual(submitted[0]['raw_payload'],'156+123+516+231')
        self.assertFalse(submitted[0]['target_pose_valid'])
        self.assertIsNone(submitted[0]['x_mm'])
        self.assertEqual(submitted[0]['image_profile']['width'],640)
        self.assertEqual(report['uart_queued_observations'],4)
        self.assertTrue(camera.closed)


if __name__=='__main__':
    unittest.main()
