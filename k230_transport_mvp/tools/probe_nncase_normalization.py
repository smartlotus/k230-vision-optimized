"""Compile tiny K230 normalization fixtures; no task-model or board claims."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.material_model_contract import compiler_settings,require_nncase_version
from tools.material_preprocess import model_uint8,float_reference


def probe(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Immutable normalization probe directory required')
    output.mkdir(parents=True)
    import onnx
    from onnx import helper,TensorProto
    versions=require_nncase_version()
    spec=importlib.util.find_spec('_nncase')
    if spec is None:raise ValueError('Missing compiler extension')
    plugin=str(Path(spec.origin).parent)
    os.environ['NNCASE_PLUGIN_PATH']=plugin
    os.environ['PATH']=plugin+os.pathsep+os.environ.get('PATH','')
    dll=os.add_dll_directory(plugin) if os.name=='nt' else None
    import nncase
    # Confine compiler generated scratch files to this new local directory.
    os.chdir(output)
    patterns={name:np.full((360,640,3),value,np.uint8) for name,value in (('zero',0),('mid',128),('max',255))}
    patterns['rgb_planes']=np.broadcast_to(np.array([32,128,224],np.uint8),(360,640,3)).copy()
    patterns['rgb_reverse']=np.broadcast_to(np.array([224,128,32],np.uint8),(360,640,3)).copy()
    rows=[]
    for size in (224,320):
        graph=helper.make_graph([helper.make_node('GlobalAveragePool',['images'],['means'])],
            'NORMALIZATION_FIXTURE_ONLY',[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,size,size])],
            [helper.make_tensor_value_info('means',TensorProto.FLOAT,[1,3,1,1])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',12)],ir_version=8)
        onnx.checker.check_model(model)
        onnx_path=output/('normalization-%d.onnx'%size)
        onnx.save(model,str(onnx_path))
        settings=compiler_settings(size)
        options=nncase.CompileOptions()
        for key,value in settings.items():setattr(options,key,value)
        options.dump_dir=str(output/('dump-%d'%size))
        compiler=nncase.Compiler(options)
        compiler.import_onnx(onnx_path.read_bytes(),nncase.ImportOptions())
        tensors=[model_uint8(rgb,size)[0] for rgb in patterns.values()]
        ptq=nncase.PTQTensorOptions();ptq.samples_count=len(tensors)
        ptq.calibrate_method='NoClip';ptq.quant_type='uint8';ptq.w_quant_type='uint8'
        ptq.set_tensor_data([[tensor] for tensor in tensors])
        compiler.use_ptq(ptq)
        start=time.perf_counter();compiler.compile();model_bytes=compiler.gencode_tobytes()
        kmodel_path=output/('normalization-%d.kmodel'%size);kmodel_path.write_bytes(model_bytes)
        simulator=nncase.Simulator();simulator.load_model(model_bytes)
        if simulator.inputs_size!=1 or simulator.outputs_size!=1:raise ValueError('Unexpected simulator interface')
        input_shape=list(simulator.get_input_shape(0))
        if input_shape!=[1,3,size,size]:raise ValueError('Compiled NCHW shape mismatch')
        comparisons=[]
        for (name,_),tensor in zip(patterns.items(),tensors):
            simulator.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(tensor))
            simulator.run()
            actual=simulator.get_output_tensor(0).to_numpy().copy()
            expected=np.mean(float_reference(tensor),axis=(2,3),keepdims=True)
            if actual.shape!=(1,3,1,1) or not np.all(np.isfinite(actual)):raise ValueError('Invalid simulator means')
            max_error=float(np.max(np.abs(actual-expected)))
            if max_error>2/255:raise ValueError('Normalization/channel mismatch')
            comparisons.append(dict(pattern=name,input_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
                expected=expected.reshape(-1).tolist(),actual=actual.reshape(-1).tolist(),max_abs_error=max_error,passed=True))
        rows.append(dict(size=size,compiler_options=settings,calibration_samples_count=ptq.samples_count,
            calibration_source='SYNTHETIC_TOOLCHAIN_FIXTURES_NOT_TASK_DATA',calibration_method='NoClip',
            onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),kmodel_sha256=hashlib.sha256(model_bytes).hexdigest(),
            kmodel_bytes=len(model_bytes),compile_and_simulate_seconds=time.perf_counter()-start,comparisons=comparisons))
        print(json.dumps({'size':size,'kmodel_bytes':len(model_bytes),'patterns_passed':len(comparisons)}),flush=True)
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='NORMALIZATION_FIXTURE_COMPILE_SIM_PASS_ONLY',
        versions=versions,models=rows,max_abs_tolerance=2/255,
        real_optimizer_updates=0,task_model_compiled=False,task_accuracy_verified=False,
        native_ai2d_equivalence_verified=False,board_run=False,kpu_execution_or_performance_verified=False,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'tools/probe_nncase_normalization.py','tools/material_model_contract.py','tools/material_preprocess.py')})
    (output/'normalization-probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if dll:dll.close()
    print(json.dumps({'status':report['status'],'models':len(rows),'board_run':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    probe(p.parse_args().output)
