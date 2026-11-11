import argparse,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from tools import manual_material_capture as capture


class ManualCaptureUnknownTests(unittest.TestCase):
    def test_ack_without_persisted_frame_retains_transaction_lock_and_cannot_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);local=root/'.ops/local';local.mkdir(parents=True)
            (local/'preview_control.py').write_text('# hostfixture only')
            (local/'canmv-active-resume.json').write_text(json.dumps({'token':'a'*32,'seq':0}))
            class FakeMcp:
                def request(self,*args):return {}
                def tool(self,name,args=None):
                    if name=='canmv_script_running':return {'running':True}
                    if name=='canmv_list_dir':return {'entries':[]}
                    raise AssertionError('Unexpected fixture tool: '+name)
            client=FakeMcp();helper={'mcp_connection':lambda *args:('HOST_FIXTURE',{}),'OfficialMcp':lambda *args:client}
            arguments=argparse.Namespace(scene='G001',output=root/'photos',camera_height_cm=None,lighting='UNMEASURED')
            ack=types.SimpleNamespace(returncode=0,stdout=json.dumps({'result':'ACK','ok':True,'seq':1}))
            with patch.object(capture,'ROOT',root),patch.object(capture.argparse.ArgumentParser,'parse_args',return_value=arguments), \
                 patch.object(capture.runpy,'run_path',return_value=helper),patch.object(capture.subprocess,'run',return_value=ack) as request, \
                 patch.object(capture.time,'monotonic',side_effect=[0,100]):
                with self.assertRaisesRegex(ValueError,'persistenceUNKNOWN'):capture.main()
                self.assertTrue((local/'manual-material-photo.lock').exists())
                with self.assertRaisesRegex(SystemExit,'capture lock exists'):capture.main()
                self.assertEqual(request.call_count,1)
            receipts=list((root/'photos/captures').glob('*.json'));self.assertEqual(len(receipts),1)
            self.assertEqual(json.loads(receipts[0].read_text())['status'],'INCOMPLETE_OR_UNKNOWN')


if __name__=='__main__':unittest.main()
