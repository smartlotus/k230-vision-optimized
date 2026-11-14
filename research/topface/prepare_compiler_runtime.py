"""Freeze Linux Python, nncase and dependencies for an offline Kaggle compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from cloud_store import atomic_json, sha


PACKAGES = {
    'nncase':'2.11.0', 'nncase-kpu':'2.11.0', 'numpy':'2.2.6',
    'opencv-python-headless':'4.11.0.86', 'onnx':'1.17.0', 'onnxruntime':'1.24.4',
    'protobuf':'5.29.5', 'flatbuffers':None, 'packaging':None, 'sympy':None,
    'coloredlogs':None, 'humanfriendly':None, 'mpmath':'1.3.0',
    'requests':'2.32.5', 'certifi':None, 'charset-normalizer':None,
    'idna':None, 'urllib3':None,
}
PLATFORMS = ('manylinux2014_x86_64','manylinux_2_17_x86_64',
             'manylinux_2_27_x86_64','manylinux_2_28_x86_64','manylinux_2_34_x86_64')


def fetch(url, path):
    result = subprocess.run(['curl.exe','--fail','--location','--silent','--show-error',
        '--retry','2','--connect-timeout','20','--max-time','600','--output',str(path),url],
        capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('Compiler dependency download failed')


def metadata(url):
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'metadata.json'
        fetch(url,path)
        return json.loads(path.read_text(encoding='utf8'))


def build(staging):
    staging=staging.resolve()
    staging.mkdir(parents=True,exist_ok=True)
    state_path=staging/'RUNTIME_MANIFEST.json'
    state=json.loads(state_path.read_text(encoding='utf8')) if state_path.exists() else {
        'schema':'topface-offline-compiler-runtime-v1.1','artifacts':[]}
    if not state['artifacts']:
        python=metadata('https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20261110')
        name='cpython-3.11.18+20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
        asset=next(item for item in python['assets'] if item['name']==name)
        state['artifacts'].append({'kind':'python','path':'python311.bin','original_name':name,
            'url':asset['browser_download_url'],'bytes':asset['size'],
            'sha256':asset['digest'].removeprefix('sha256:')})
        dotnet=metadata('https://dotnetcli.blob.core.windows.net/dotnet/release-metadata/7.0/releases.json')
        release=next(item for item in dotnet['releases'] if item['runtime']['version']=='7.0.20')
        package=next(item for item in release['runtime']['files'] if item['rid']=='linux-x64' and item['name'].endswith('.tar.gz'))
        state['artifacts'].append({'kind':'dotnet','path':'dotnet7.bin','original_name':package['name'],
                                   'url':package['url'],'sha512':package['hash']})
        tags=set(cpython_tags((3,11),abis=['cp311'],platforms=PLATFORMS))
        tags.update(compatible_tags((3,11),interpreter='cp311',platforms=PLATFORMS))
        for package,version in PACKAGES.items():
            url=f'https://pypi.org/pypi/{package}/'+(version+'/' if version else '')+'json'
            info=metadata(url)
            candidates=[item for item in info['urls'] if item['filename'].endswith('.whl')
                        and tags.intersection(parse_wheel_filename(item['filename'])[-1])]
            if not candidates:
                raise ValueError('No compatible Linux wheel for '+package)
            selected=sorted(candidates,key=lambda item:item['filename'])[0]
            state['artifacts'].append({'kind':'wheel','package':package,'version':info['info']['version'],
                'path':package+'.bin','original_name':selected['filename'],'url':selected['url'],
                'bytes':selected['size'],'sha256':selected['digests']['sha256'],
                'requires_dist':info['info']['requires_dist']})
        atomic_json(state_path,state)
    for item in state['artifacts']:
        path=staging/item['path']
        if not path.exists():
            temporary=path.with_suffix('.part')
            fetch(item['url'],temporary)
            temporary.replace(path)
        if 'bytes' in item and path.stat().st_size!=item['bytes']:
            raise ValueError('Compiler runtime size differs')
        if 'sha512' in item:
            if hashlib.sha512(path.read_bytes()).hexdigest()!=item['sha512']:
                raise ValueError('Official .NET archive digest differs')
        if 'sha256' in item and sha(path)!=item['sha256']:
            raise ValueError('Compiler runtime digest differs')
        item.update(bytes=path.stat().st_size,sha256=sha(path))
        atomic_json(state_path,state)
        print(json.dumps({'artifact':item['original_name'],'bytes':item['bytes'],'verified':True}),flush=True)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--staging',type=Path,required=True)
    build(**vars(parser.parse_args()))
