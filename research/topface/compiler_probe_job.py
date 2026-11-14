"""Qualify the frozen offline Linux nncase toolchain on a CPU Kaggle job."""
import json
from pathlib import Path
import subprocess
import time

from cloud_store import archive_directory,atomic_json,sha
from compiler_runtime import install
from native_job import verify_inputs


def run(input_root,run_id):
    input_root=Path(input_root).resolve()
    verify_inputs(input_root)
    started=time.monotonic()
    root=Path('/kaggle/working')
    scratch=Path('/kaggle/temp') if Path('/kaggle/temp').exists() else Path('/tmp')
    output=root/run_id
    output.mkdir(exist_ok=False)
    state={'run_id':run_id,'status':'STARTED','optimizer_updates':0,
           'input_manifest_sha256':sha(input_root/'INPUT_MANIFEST.json')}
    try:
        python,env=install(input_root,scratch/run_id)
        print('TOPFACE_COMPILER_STAGE RUNTIME_READY',flush=True)
        with (output/'probe.log').open('wb') as log:
            process=subprocess.run([str(python),str(input_root/'probe_rgb_preprocess.py'),
                '--output',str(output/'fixture')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if process.returncode:
            raise RuntimeError('Linux compiler fixture failed')
        report=json.loads((output/'fixture/RESULT.json').read_text(encoding='utf8'))
        if report['status']!='FIXTURE_COMPILE_AND_SEMANTICS_PASS':
            raise ValueError('Linux compiler preprocessing semantics differ')
        state.update(status='LINUX_COMPILER_FIXTURE_PASS',compiler=report['compiler'],
                     comparisons=report['comparisons'])
    except Exception as error:
        state.update(status='LINUX_COMPILER_FIXTURE_FAILED',error_type=type(error).__name__)
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'STATE.json',state)
        archive=root/(run_id+'.tar.gz')
        archive_directory(output,archive)
        state.update(archive={'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},
                     durable_status='PUBLISHER_PENDING_EXACT_SAVED_OUTPUT')
        atomic_json(root/(run_id+'-RESULT.json'),state)
        print('TOPFACE_COMPILER_RESULT '+json.dumps(state),flush=True)
    return state
