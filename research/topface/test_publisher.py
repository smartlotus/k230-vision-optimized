"""Offline regression of credential-boundary and full-byte proof behavior."""
import hashlib
import io
import json
import time
import unittest
from publisher.ci_transfer import parse_spec,TransferError,verify_body,readback


def spec():
    digest='a'*64;run='k230-publisher-probe-a01'
    return {'run_id':run,'source_url':'https://www.kaggleusercontent.com/placeholder/archive.tar.gz',
        'upload_url':'https://www.googleapis.com/upload/drive/v3/files?upload_id=placeholder',
        'drive_access_token':'PLACEHOLDER_NOT_A_REAL_TOKEN_'+'X'*24,'drive_expiry_epoch':time.time()+3600,
        'archive':{'name':run+'-'+digest[:16]+'.tar.gz','bytes':4,'sha256':digest},
        'folder_id':'1MwltNmaI24HjvcxnG1w608qui42RAPKf','drive_file_id':'placeholder_file_id',
        'prep':{'ref':'lotus30/k230-topface-v11-02-dataset-audit','kernel_id':145013738,
            'version':3,'provider_status':'COMPLETE','source_sha256':'b'*64}}


class Response(io.BytesIO):
    def __init__(self,data,status=200):super().__init__(data);self.status=status;self.headers={'Content-Length':str(len(data))}


class PublisherTests(unittest.TestCase):
    def test_only_fixed_folder_allowlisted_source_and_short_access_envelope(self):
        value=spec();self.assertEqual(parse_spec(json.dumps(value))['run_id'],value['run_id'])
        for field,replacement in (('folder_id','different_folder'),('source_url','https://example.org/archive')):
            bad=spec();bad[field]=replacement
            with self.assertRaises(TransferError):parse_spec(json.dumps(bad))
        bad=spec();bad['refresh_token']='forbidden-longterm-placeholder'
        with self.assertRaises(TransferError):parse_spec(json.dumps(bad))
        bad=spec();bad['drive_expiry_epoch']=time.time()-1
        with self.assertRaises(TransferError):parse_spec(json.dumps(bad))

    def test_entire_body_hash_required_and_last_byte_tampering_fails(self):
        data=b'full-original-bytes';archive={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        verify_body(Response(data),archive)
        with self.assertRaises(TransferError):verify_body(Response(data[:-1]+b'!'),archive)
        with self.assertRaises(TransferError):verify_body(Response(data[:-1]),archive)

    def test_readback_checks_destination_and_stream_bytes(self):
        value=spec();data=b'data';value['archive']['sha256']=hashlib.sha256(data).hexdigest()
        value['archive']['name']=value['run_id']+'-'+value['archive']['sha256'][:16]+'.tar.gz'
        meta={'id':value['drive_file_id'],'name':value['archive']['name'],'parents':[value['folder_id']],
              'size':'4','trashed':False,'sha256Checksum':value['archive']['sha256']}
        class Opener:
            def __init__(self,body):self.body=body;self.urls=[]
            def open(self,request,timeout):
                self.urls.append(request.full_url)
                return Response(self.body if 'alt=media' in request.full_url else json.dumps(meta).encode())
        opener=Opener(data);readback(opener,value);self.assertEqual(len(opener.urls),2)
        with self.assertRaises(TransferError):readback(Opener(b'FAIL'),value)
        meta['parents']=['different_folder']
        with self.assertRaises(TransferError):readback(Opener(data),value)


if __name__=='__main__':unittest.main()
