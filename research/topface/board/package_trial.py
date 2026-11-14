"""Build the standalone CanMV TopFaceNet evaluation bundle."""
import argparse
import json
from pathlib import Path
import shutil

from cloud_store import atomic_json, sha


HERE = Path(__file__).resolve().parent


def package(compiled, export, output):
    identity = json.loads((compiled/'identity.json').read_text(encoding='utf8'))
    result = json.loads((compiled/'RESULT.json').read_text(encoding='utf8'))
    pt = json.loads((export/'RESULT.json').read_text(encoding='utf8'))
    contract = json.loads((compiled/'output_contract.json').read_text(encoding='utf8'))
    if pt['experiment'] != 'formal_full' or result['identity'] != identity:
        raise ValueError('Board package requires the frozen formal model')
    model = compiled/'topface.kmodel'
    if sha(model) != identity['kmodel_sha256'] or pt['onnx_sha256'] != identity['onnx_sha256']:
        raise ValueError('Compiled model and floating-point reference differ')
    output.mkdir(parents=True,exist_ok=False)
    for source in (HERE/'topface_trial.py', HERE/'topface_decode.py', model):
        shutil.copy2(source, output/source.name)
    atomic_json(output/'identity.json', identity)
    atomic_json(output/'output_contract.json', contract)
    (output/'README.md').write_text(
        '# K230 TopFaceNet CSI2 evaluation\n\n'
        'Copy this directory to `/sdcard/topface_trial` and run the following in CanMV IDE.\n\n'
        '```python\nimport sys\nsys.path.insert(0, "/sdcard/topface_trial")\n'
        'from topface_trial import run\nrun(frames=600, display=True)\n```\n\n'
        'The serial report records KPU, AI2D and full camera-to-display timing, capture timeouts and sampled memory. '
        'Use the recorded runtime version and compare the model SHA-256 before interpreting a run. '
        'For the longer stability run, call `run(frames=18000, display=True)`.\n\n'
        'References: [CanMV AI manual](https://www.kendryte.com/k230_canmv/en/main/ai_dev_doc.html), '
        '[nncase toolchain](https://github.com/kendryte/nncase/blob/master/docs/USAGE_v2.md).\n',
        encoding='utf8')
    manifest = {'status':'DEPLOYMENT_BUNDLE_READY','board_test':'NOT_RUN',
                'files':[{'name':file.name,'bytes':file.stat().st_size,'sha256':sha(file)}
                         for file in sorted(output.iterdir())]}
    atomic_json(output/'MANIFEST.json',manifest)
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for key in ('compiled','export','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(package(**vars(parser.parse_args())),indent=2))
