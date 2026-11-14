"""Consume a fixed archive on the cloud producer; no host artifact relay."""
import hashlib
import json
from pathlib import Path
import tarfile

from cloud_store import API,sha


def fetch_archive(drive,receipt,destination,max_bytes=1024**3):
    destination=Path(destination)
    if receipt['status']!='FULL_BYTE_SHA256_PASS' or receipt['folder_id']!=drive.folder_id:
        raise ValueError('Unaccepted archive or different destination')
    size=receipt['bytes'];digest=receipt['sha256']
    if not 0<size<=max_bytes:raise ValueError('Cloud archive size outside registered cap')
    with drive.request('GET',API+'/'+receipt['file_id'],params={'fields':'id,name,size,parents,trashed'}) as response:
        meta=response.json()
    if meta.get('trashed') or meta.get('parents')!=[drive.folder_id] or meta.get('name')!=receipt['name'] or int(meta.get('size',-1))!=size:
        raise ValueError('Drive source archive identity changed')
    digestor=hashlib.sha256();count=0
    with drive.request('GET',API+'/'+receipt['file_id'],params={'alt':'media'},stream=True) as response:
        with destination.open('xb') as stream:
            for chunk in response.iter_content(1024**2):
                drive.check_budget();count+=len(chunk)
                if count>size:raise ValueError('Archive stream exceeds frozen source size')
                stream.write(chunk);digestor.update(chunk)
    if (count,digestor.hexdigest())!=(size,digest):raise ValueError('Cloud source archive SHA mismatch')
    return destination


def extract_verified_archive(archive,destination,max_members=20000,max_expanded_bytes=3*1024**3):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or len(members)>max_members or sum(m.size for m in members)>max_expanded_bytes:
            raise ValueError('Archive inventory ambiguous or exceeds admission bounds')
        for member in members:
            path=(destination/member.name).resolve()
            if not member.isfile() or not path.is_relative_to(destination) or '\\' in member.name:
                raise ValueError('Archive contains unsafe member')
        manifest_member=next((m for m in members if m.name=='MEMBERS.json'),None)
        if manifest_member is None:raise ValueError('Cloud archive has no member inventory')
        raw=bundle.extractfile(manifest_member).read()
        manifest=json.loads(raw)
        expected={r['name']:r for r in manifest}
        if len(expected)!=len(manifest) or set(names)!=(set(expected)|{'MEMBERS.json'}):
            raise ValueError('Archive member denominator differs from manifest')
        bundle.extractall(destination,members=members,filter='data')
    for name,row in expected.items():
        path=destination/name
        if (path.stat().st_size,sha(path))!=(row['bytes'],row['sha256']):
            raise ValueError('Extracted member hash mismatch')
    return {'files':len(expected),'expanded_bytes':sum(r['bytes'] for r in expected.values()),
        'member_manifest_sha256':hashlib.sha256(raw).hexdigest(),'sha256_verification':'ALL_MEMBERS_PASS'}
