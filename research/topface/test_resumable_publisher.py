"""Recover exact byte offsets without allocating a second Drive object."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from publisher.ci_transfer import upload_archive,transfer,TransferError


class Response(io.BytesIO):
    def __init__(self,status,body=b'',headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}


class ResumablePublisherTests(unittest.TestCase):
    def fixture(self):
        return {'archive':{'bytes':12},'drive_file_id':'same-reserved-id',
                'drive_expiry_epoch':time.time()+3600,'upload_url':'unused-placeholder'}

    def test_queries_original_session_then_sends_only_remaining_bytes(self):
        calls=[]
        def request(_,url,**options):
            calls.append(options)
            if len(calls)==1:return Response(308,headers={'Range':'bytes=0-7'})
            return Response(200,json.dumps({'id':'same-reserved-id'}).encode())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'archive';path.write_bytes(b'abcdefghijkl')
            with patch('publisher.ci_transfer.request',side_effect=request),patch('publisher.ci_transfer.CHUNK_BYTES',4):
                upload_archive(None,self.fixture(),path)
        self.assertEqual(calls[0]['headers']['Content-Range'],'bytes */12')
        self.assertEqual(calls[1]['headers']['Content-Range'],'bytes 8-11/12')
        self.assertEqual(calls[1]['data'],b'ijkl')
        self.assertEqual(len(calls),2)

    def test_nonprogressing_session_has_a_finite_retry_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'archive';path.write_bytes(b'abcdefghijkl')
            with patch('publisher.ci_transfer.request',side_effect=lambda *a,**k:Response(308)) as request, \
                 patch('publisher.ci_transfer.CHUNK_BYTES',4),patch('publisher.ci_transfer.time.sleep'):
                with self.assertRaisesRegex(TransferError,'needs_reconciliation'):
                    upload_archive(None,self.fixture(),path)
                self.assertLessEqual(request.call_count,10)

    def test_fresh_session_still_uploads_every_chunk_in_order(self):
        calls=[]
        def request(_,url,**options):
            calls.append(options['headers']['Content-Range'])
            if len(calls)==1:return Response(308)
            if len(calls)<4:return Response(308,headers={'Range':'bytes=0-'+str((len(calls)-1)*4-1)})
            return Response(200,json.dumps({'id':'same-reserved-id'}).encode())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'archive';path.write_bytes(b'abcdefghijkl')
            with patch('publisher.ci_transfer.request',side_effect=request),patch('publisher.ci_transfer.CHUNK_BYTES',4):
                upload_archive(None,self.fixture(),path)
        self.assertEqual(calls,['bytes */12','bytes 0-3/12','bytes 4-7/12','bytes 8-11/12'])

    def test_completed_object_goes_to_full_readback_without_upload(self):
        import hashlib
        body=b'abcdefghijkl'
        spec={**self.fixture(),'source_url':'unused-placeholder','drive_access_token':'unused-placeholder',
              'archive':{'bytes':12,'sha256':hashlib.sha256(body).hexdigest(),'name':'unchanged'},
              'run_id':'k230-fixture-a01','folder_id':'unchanged','prep':{'version':1}}
        with patch('publisher.ci_transfer.source_response',return_value=Response(200,body)), \
             patch('publisher.ci_transfer.request',return_value=Response(200)), \
             patch('publisher.ci_transfer.upload_archive') as upload,patch('publisher.ci_transfer.readback') as readback:
            transfer(spec,None)
            upload.assert_not_called();readback.assert_called_once()


if __name__=='__main__':unittest.main()
