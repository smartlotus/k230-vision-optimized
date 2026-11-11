"""Reviewed task ONNX -> nncase2.11 K230 conversion, with immutable input checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.material_dataset import audit,canonical
from tools.material_model_contract import (inspect_onnx,compiler_settings,load_reviewed_calibration,
                                           require_nncase_version,verify_training_provenance)


def preflight(root,manifest,onnx_path,calibration,size):
    dataset=audit(root,manifest)
    if not dataset['export_ready']:raise ValueError('Reviewed grouped train/val/test and positive training objects required')
    model=inspect_onnx(onnx_path,size)  # Default requires TRAINED, never random probe.
    arrays,receipt=load_reviewed_calibration(root,manifest,calibration,size)
    plan=dict(schema_version=1,status='PREFLIGHT_ONLY_NOT_COMPILED',dataset_manifest_sha256=dataset['manifest_sha256'],
              model=model,compiler_options=compiler_settings(size),required_nncase='2.11.0',required_nncase_kpu='2.11.0',
              samples_count=len(arrays),calibration_manifest_sha256=hashlib.sha256(canonical(receipt)).hexdigest(),
              calibration_image_ids=[r['image_id'] for r in receipt['samples']],
              calibration_tensor_sha256=[r['sha256'] for r in receipt['samples']],
              quantization=dict(calibrate_method='NoClip',quant_type='uint8',w_quant_type='uint8'),
              compiler_executed=False,task_accuracy_verified=False,native_ai2d_equivalence_verified=False)
    return plan,arrays


def compile_model(root,manifest,onnx_path,calibration,size,output,manifest_path=None,training_run=None):
    """No dataset acquisition or training; all gates run before output creation."""
    root=Path(root).resolve();calibration=Path(calibration).resolve();onnx_path=Path(onnx_path).resolve()
    if manifest_path is None:raise ValueError('Persistent reviewed manifest path required')
    manifest_path=Path(manifest_path).resolve()
    if canonical(json.loads(manifest_path.read_text(encoding='utf-8')))!=canonical(manifest):
        raise ValueError('Persistent reviewed manifest changed before conversion')
    plan,arrays=preflight(root,manifest,onnx_path,calibration,size)
    plan['training_provenance']=verify_training_provenance(root,manifest,onnx_path,training_run)
    versions=require_nncase_version()
    output=Path(output).resolve()
    if output.exists():raise ValueError('Conversion output is immutable; choose a new path')
    model_bytes=onnx_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest()!=plan['model']['model_sha256']:
        raise ValueError('ONNX changed after preflight')
    output.mkdir(parents=True)
    previous=Path.cwd();dll=None
    previous_env={key:os.environ.get(key) for key in ('PATH','NNCASE_PLUGIN_PATH')}
    receipt=dict(plan,versions=versions,started_at=datetime.now(timezone.utc).isoformat(),status='COMPILER_STARTED',
                 stage='INITIALIZING',compiler_compile_called=False,compile_completed=False)
    (output/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    try:
        spec=importlib.util.find_spec('_nncase')
        if spec is None:raise ValueError('Missing compiler extension')
        plugin=str(Path(spec.origin).parent)
        os.environ['NNCASE_PLUGIN_PATH']=plugin
        os.environ['PATH']=plugin+os.pathsep+os.environ.get('PATH','')
        dll=os.add_dll_directory(plugin) if os.name=='nt' else None
        import nncase
        os.chdir(output)
        options=nncase.CompileOptions()
        for key,value in plan['compiler_options'].items():setattr(options,key,value)
        options.dump_dir=str(output/'compiler-dump')
        compiler=nncase.Compiler(options)
        compiler.import_onnx(model_bytes,nncase.ImportOptions())
        receipt['stage']='MODEL_IMPORTED'
        ptq=nncase.PTQTensorOptions();ptq.samples_count=plan['samples_count']
        for key,value in plan['quantization'].items():setattr(ptq,key,value)
        ptq.set_tensor_data(arrays);compiler.use_ptq(ptq)
        receipt.update(stage='COMPILING',compiler_compile_called=True)
        (output/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        compiler.compile();receipt.update(stage='COMPILE_RETURNED',compile_completed=True,compiler_executed=True)
        kmodel=compiler.gencode_tobytes()
        if not kmodel:raise ValueError('Empty kmodel')
        # Data/review/splits and source bytes must still match after compilation.
        confirmed,_=preflight(root,manifest,onnx_path,calibration,size)
        confirmed['training_provenance']=verify_training_provenance(root,manifest,onnx_path,training_run)
        if (confirmed!=plan or canonical(json.loads(manifest_path.read_text(encoding='utf-8')))!=canonical(manifest)):
            raise ValueError('Conversion inputs changed while compiling')
        (output/'material.kmodel').write_bytes(kmodel)
        receipt.update(status='COMPILED_NOT_TASK_ACCEPTED',stage='GENERATED_AND_INPUTS_RECHECKED',compiler_executed=True,kmodel_bytes=len(kmodel),
                       kmodel_sha256=hashlib.sha256(kmodel).hexdigest(),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(status='FAILED_OR_UNKNOWN_NO_ACCEPTANCE',error=str(error))
        raise
    finally:
        try:(output/'conversion.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        finally:
            os.chdir(previous)
            if dll:dll.close()
            for key,value in previous_env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','compile'));p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--size',type=int,choices=(224,320),required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--training-run',type=Path)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    try:
        if a.action=='preflight':result,_=preflight(a.root.resolve(),manifest,a.model,a.calibration.resolve(),a.size)
        else:
            if a.output is None:raise ValueError('Explicit new output path required')
            result=compile_model(a.root.resolve(),manifest,a.model,a.calibration.resolve(),a.size,a.output,a.manifest,a.training_run)
        print(json.dumps(result,indent=2))
    except (ValueError,OSError) as error:
        print(json.dumps(dict(status='NOT_ADMITTED',reason=str(error),task_accuracy_verified=False)))
        raise SystemExit(2)
