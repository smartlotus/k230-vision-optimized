"""Fault injection validates diagnostic cleanup, not KPU recognition."""
import gc,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools.kpu_bounded_probe import probe


class KpuProbeCleanupTests(unittest.TestCase):
    def test_failed_load_preserves_error_and_attempts_every_cleanup_stage(self):
        calls=[]
        class Kpu:
            def load_kmodel(self,path):raise RuntimeError('original load failure')
            def __getattribute__(self,name):
                if name=='__del__':return destroy
                return object.__getattribute__(self,name)
        def destroy():calls.append('destroy');raise RuntimeError('destroy failure')
        def shrink():calls.append('shrink');raise RuntimeError('shrink failure')
        runtime=types.SimpleNamespace(version=lambda:'FAKE_HOST_ONLY',kpu=Kpu,shrink_memory_pool=shrink)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'example.kmodel';path.write_bytes(b'fake fixture')
            with patch.dict('sys.modules',{'nncase_runtime':runtime,'ulab':types.SimpleNamespace(numpy=np),'ulab.numpy':np}), \
                 patch.object(time,'ticks_us',return_value=1,create=True), \
                 patch.object(gc,'collect',side_effect=lambda:calls.append('gc')):
                result=probe(str(path),iterations=1)
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['error'],'original load failure')
        self.assertEqual(calls,['destroy','shrink','gc'])
        self.assertEqual(len(result['cleanup_errors']),2)
        self.assertFalse(result['recognition_verified'])


if __name__=='__main__':unittest.main()
