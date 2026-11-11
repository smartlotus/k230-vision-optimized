import unittest
from k230.exposure_control import configure_exposure


class Sensor:
    def __init__(self,value=12345):self.auto=True;self.value=value;self.writes=[]
    def auto_exposure(self,value=None):
        if value is not None:self.auto=value;self.writes.append(('auto',value))
        return self.auto
    def exposure(self,value=None):
        if value is not None:self.value=value;self.writes.append(('exposure',value))
        return self.value


class ExposureTests(unittest.TestCase):
    def test_auto_mode_only_reads_and_lock_reuses_measured_value(self):
        sensor=Sensor()
        self.assertFalse(configure_exposure(sensor)['locked'])
        self.assertEqual(sensor.writes,[])
        result=configure_exposure(sensor,True)
        self.assertTrue(result['locked'])
        self.assertEqual(sensor.writes,[('auto',False),('exposure',12345)])
        self.assertEqual(result['exposure_after_us'],12345)

    def test_unavailable_exposure_is_not_replaced_with_a_guessed_value(self):
        sensor=Sensor(None)
        result=configure_exposure(sensor,True)
        self.assertFalse(result['locked'])
        self.assertIsNotNone(result['error'])
        self.assertNotIn(('auto',False),sensor.writes)
        self.assertTrue(sensor.auto)


if __name__=='__main__':unittest.main()
