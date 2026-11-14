"""Tiny compiler RGBuint8 -> official NanoDet BGRmean/std semantic probe.

Not the task model, final PTQ, native AI2D equivalence, or a board benchmark.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

from cloud_store import atomic_json,sha
from task_geometry import MEAN,STD


def main(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    import onnx
    from onnx import helper,TensorProto
    spec=importlib.util.find_spec('_nncase');plugin=Path(spec.origin).parent
    os.environ['NNCASE_PLUGIN_PATH']=str(plugin);os.environ['PATH']=str(plugin)+os.pathsep+os.environ['PATH']
    dll=os.add_dll_directory(str(plugin)) if os.name=='nt' else None
    import nncase
    os.chdir(output)
    height,width=128,224
    graph=helper.make_graph([helper.make_node('GlobalAveragePool',['image'],['bgr_means'])],
        'RGB_UINT8_BGR_MEANSTD_ENGINEERING_ONLY',
        [helper.make_tensor_value_info('image',TensorProto.FLOAT,[1,3,height,width])],
        [helper.make_tensor_value_info('bgr_means',TensorProto.FLOAT,[1,3,1,1])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',11)],ir_version=8)
    onnx.checker.check_model(model);onnx.save(model,str(output/'fixture.onnx'))
    options=nncase.CompileOptions()
    settings={'target':'k230','preprocess':True,'input_shape':[1,3,height,width],
        'input_type':'uint8','input_range':[0,255],'swapRB':True,'mean':MEAN.tolist(),
        'std':STD.tolist(),'input_layout':'NCHW','output_layout':'','shape_bucket_enable':False,
        'dump_ir':True,'dump_asm':False,'dump_dir':str(output/'compiler-dump')}
    for key,value in settings.items():setattr(options,key,value)
    patterns={'zero':(0,0,0),'mid':(128,128,128),'max':(255,255,255),
              'rgb_planes':(32,128,224),'rgb_reverse':(224,128,32)}
    tensors=[np.broadcast_to(np.array(rgb,np.uint8)[None,:,None,None],(1,3,height,width)).copy() for rgb in patterns.values()]
    compiler=nncase.Compiler(options);compiler.import_onnx(model.SerializeToString(),nncase.ImportOptions())
    ptq=nncase.PTQTensorOptions();ptq.samples_count=len(tensors)
    ptq.calibrate_method='NoClip';ptq.quant_type='uint8';ptq.w_quant_type='uint8'
    ptq.set_tensor_data([[tensor] for tensor in tensors]);compiler.use_ptq(ptq)
    compiler.compile();kmodel=compiler.gencode_tobytes();(output/'fixture.kmodel').write_bytes(kmodel)
    sim=nncase.Simulator();sim.load_model(kmodel);comparisons=[]
    for (name,rgb),tensor in zip(patterns.items(),tensors):
        sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(tensor));sim.run()
        actual=sim.get_output_tensor(0).to_numpy().reshape(-1).astype(np.float32)
        expected=(np.array(rgb[::-1],np.float32)-MEAN)/STD
        error=float(np.max(np.abs(actual-expected)))
        comparisons.append({'pattern':name,'rgb_uint8':rgb,'expected_bgr_normalized':expected.tolist(),
            'simulator':actual.tolist(),'max_abs_error':error,'pass':bool(np.isfinite(actual).all() and error<=.06)})
    result={'schema':'topface-rgb-preprocess-fixture-v1.1','observed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'FIXTURE_COMPILE_AND_SEMANTICS_PASS' if all(r['pass'] for r in comparisons) else 'SEMANTICS_FAILED',
        'compiler':{p:importlib.metadata.version(p) for p in ('nncase','nncase-kpu')},
        'settings':settings,'comparisons':comparisons,'max_abs_tolerance':.06,
        'fixture_onnx_sha256':sha(output/'fixture.onnx'),'fixture_kmodel_sha256':sha(output/'fixture.kmodel'),
        'task_model_compiled':False,'optimizer_updates':0,'board_run':False,'native_ai2d_pixel_equivalence':False,
        'final_ptq':False,'scope':'small toolchain fixture only; task raw and decoded outputs must be reverified after final training'}
    atomic_json(output/'RESULT.json',result)
    if dll:dll.close()
    print(json.dumps(result,indent=2))
    if result['status']!='FIXTURE_COMPILE_AND_SEMANTICS_PASS':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
