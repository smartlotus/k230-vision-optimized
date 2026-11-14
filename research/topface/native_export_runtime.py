"""Freeze and install ONNX dependencies for Kaggle's native CPython 3.13."""
import json
from pathlib import Path
import shutil
import sys
import zipfile

from cloud_store import atomic_json, sha


PACKAGES={'onnx':'1.18.0','onnxruntime':'1.24.4','protobuf':'5.29.5',
          'typing-extensions':'4.15.0','flatbuffers':'25.12.19','packaging':'25.0',
          'sympy':'1.14.0','mpmath':'1.3.0','coloredlogs':'15.0.1','humanfriendly':'10.0'}


def build(destination):
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import parse_wheel_filename
    from prepare_compiler_runtime import PLATFORMS, metadata, fetch
    destination.mkdir(parents=True,exist_ok=True)
    manifest=destination/'NATIVE_EXPORT_RUNTIME.json'
    if manifest.exists():
        state=json.loads(manifest.read_text(encoding='utf8'))
    else:
        tags=set(cpython_tags((3,13),abis=['cp313'],platforms=PLATFORMS))
        tags.update(compatible_tags((3,13),interpreter='cp313',platforms=PLATFORMS))
        rows=[]
        for package,version in PACKAGES.items():
            info=metadata(f'https://pypi.org/pypi/{package}/{version}/json')
            candidates=[row for row in info['urls'] if row['filename'].endswith('.whl')
                        and tags.intersection(parse_wheel_filename(row['filename'])[-1])]
            if not candidates:
                raise ValueError('No pinned CPython 3.13 wheel for '+package)
            row=sorted(candidates,key=lambda value:value['filename'])[0]
            rows.append({'path':'native-'+package+'.bin','package':package,'version':version,
                'original_name':row['filename'],'url':row['url'],'bytes':row['size'],
                'sha256':row['digests']['sha256'],'requires_dist':info['info']['requires_dist']})
        state={'python':[3,13],'schema':'topface-native-export-runtime-v1.1','artifacts':rows}
        atomic_json(manifest,state)
    for row in state['artifacts']:
        path=destination/row['path']
        if not path.exists():
            temporary=path.with_suffix('.part')
            fetch(row['url'],temporary)
            temporary.replace(path)
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Native export wheel differs from official bytes')
    return state


def install(input_root,destination):
    manifest=json.loads((input_root/'NATIVE_EXPORT_RUNTIME.json').read_text(encoding='utf8'))
    if list(sys.version_info[:2])!=manifest['python']:
        raise ValueError('Native export wheel ABI differs from the Kaggle interpreter')
    destination.mkdir(parents=True,exist_ok=False)
    for row in manifest['artifacts']:
        path=input_root/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Native export wheel identity differs')
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist())>256*1024**2:
                raise ValueError('Native export wheel exceeds expanded size limit')
            for item in archive.infolist():
                parts=Path(item.filename).parts
                if not parts or '..' in parts or Path(item.filename).is_absolute():
                    raise ValueError('Unsafe native export wheel member')
                if item.is_dir():
                    continue
                if parts[0].endswith('.data'):
                    if len(parts)<3 or parts[1] not in ('purelib','platlib'):
                        continue
                    parts=parts[2:]
                target=destination.joinpath(*parts)
                if target.exists():
                    raise ValueError('Native export wheels collide')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source,target.open('wb') as output:
                    shutil.copyfileobj(source,output)
    sys.path.insert(0,str(destination))
    return {row['package']:row['version'] for row in manifest['artifacts']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
