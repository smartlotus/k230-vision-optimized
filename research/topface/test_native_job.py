"""Regression checks for actual cloud admission/failure recovery hazards."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import zipfile

from cloud_store import Drive, sha
from native_job import main


class NativeAdmissionTests(unittest.TestCase):
    def test_existing_drive_run_refuses_fresh_session_replay(self):
        drive=object.__new__(Drive);drive.folder_id='frozen-folder'
        response=Mock();response.json.return_value={'files':[{'id':'existing-run'}]}
        context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
        drive.request=Mock(return_value=context)
        with self.assertRaisesRegex(RuntimeError,'already admitted'):
            drive.reserve_run('pilot-a02','a'*64)
        self.assertEqual(drive.request.call_count,1)
        self.assertEqual(drive.request.call_args.args[0],'GET')

    def test_expired_budget_stops_before_network_or_large_hash(self):
        drive=object.__new__(Drive);drive.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):drive.request('GET','https://www.googleapis.com/drive/v3/files')
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'data';file.write_bytes(b'x')
            with self.assertRaises(TimeoutError):sha(file,time.monotonic()-1)

    def test_failed_render_and_archive_still_persist_compact_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'input';inputs.mkdir();work=root/'work';work.mkdir()
            (inputs/'INPUT_MANIFEST.json').write_text(json.dumps({'files':[]}),encoding='utf8')
            (inputs/'SMOKE_HUMAN_REVIEW.json').write_text(json.dumps({
                'status':'PASS','reviewed_images':30,'pilot_authorized':True}),encoding='utf8')
            with zipfile.ZipFile(inputs/'cad_bundle.bin','w') as z:z.writestr('material_block.glb',b'fixture')
            dummy_blender=root/'blender';dummy_blender.write_bytes(b'not executed in this failure fixture')
            with patch('native_job.shutil.disk_usage',return_value=Mock(free=10*1024**3)), \
                 patch('native_job.render',side_effect=RuntimeError('render failed')) as renderer, \
                 patch('native_job.archive_directory',side_effect=OSError('archive failed')):
                with self.assertRaises(RuntimeError):
                    main(inputs,work,'fixture-failure',count=6,local_blender=dummy_blender,max_seconds=600)
            result=json.loads((work/'fixture-failure-RESULT.json').read_text(encoding='utf8'))
            self.assertEqual(result['status'],'FAILED_OR_BUDGET_STOP')
            self.assertEqual(result['durable_status'],'UNACCEPTED_PRESERVE_SAVED_OUTPUT')
            self.assertEqual(result['publication_error_type'],'OSError')
            self.assertEqual(renderer.call_count,2)  # one qualification + one CPU fallback; no Pilot
            self.assertFalse(result['automatic_replay'])


if __name__=='__main__':unittest.main()
