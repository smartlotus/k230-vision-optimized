"""Install the frozen Linux compiler within a Kaggle scratch directory."""
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

from cloud_store import sha


def install(input_root, destination):
    input_root=Path(input_root).resolve()
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((input_root/'RUNTIME_MANIFEST.json').read_text(encoding='utf8'))
    site=destination/'site'
    site.mkdir()
    for row in manifest['artifacts']:
        path=(input_root/row['path']).resolve()
        if (not path.is_relative_to(input_root) or path.stat().st_size!=row['bytes'] or
                sha(path)!=row['sha256']):
            raise ValueError('Compiler runtime input differs from the frozen manifest')
        if row['kind'] in ('python','dotnet'):
            folder=destination/row['kind']
            folder.mkdir()
            with tarfile.open(path,'r:gz') as archive:
                if sum(item.size for item in archive.getmembers())>1024**3:
                    raise ValueError('Runtime archive exceeds expected size')
                archive.extractall(folder,filter='data')
        elif row['kind']=='wheel':
            with zipfile.ZipFile(path) as wheel:
                members=wheel.infolist()
                if sum(item.file_size for item in members)>512*1024**2:
                    raise ValueError('Compiler wheel exceeds expected expanded size')
                for item in members:
                    parts=Path(item.filename).parts
                    if not parts or '..' in parts or Path(item.filename).is_absolute():
                        raise ValueError('Unsafe compiler wheel member')
                    if item.is_dir():
                        continue
                    if parts[0].endswith('.data'):
                        if len(parts)<3 or parts[1] not in ('purelib','platlib'):
                            continue
                        parts=parts[2:]
                    target=site.joinpath(*parts)
                    if target.exists():
                        raise ValueError('Compiler wheels contain colliding files')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with wheel.open(item) as source,target.open('wb') as output:
                        shutil.copyfileobj(source,output)
                    mode=(item.external_attr>>16)&0o777
                    if mode:
                        target.chmod(mode)
        else:
            raise ValueError('Unknown compiler runtime artifact')
    python=destination/'python/python/bin/python3.11'
    if not python.is_file():
        raise ValueError('Standalone Python executable is missing')
    env=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','NNCASE_PLUGIN_PATH'):
        env.pop(key,None)
    env.update(PYTHONPATH=str(input_root)+os.pathsep+str(site),
               DOTNET_ROOT=str(destination/'dotnet'),DOTNET_MULTILEVEL_LOOKUP='0',
               DOTNET_SYSTEM_GLOBALIZATION_INVARIANT='1',
               NNCASE_PLUGIN_PATH=str(site),OMP_NUM_THREADS='2')
    return python,env
