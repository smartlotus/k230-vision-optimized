"""Compile actual trained exports using train-only image windows for PTQ."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from dataset import audit, build_sequences, digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--onnx',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--window',type=int,default=1)
    args=ap.parse_args();model=args.onnx.resolve();data=args.data.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    plugin=str(Path(importlib.util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH']=plugin;os.environ['PATH']=plugin+os.pathsep+os.environ.get('PATH','')
    dll=os.add_dll_directory(plugin) if os.name=='nt' else None
    import nncase
    import onnxruntime as ort
    versions={n:importlib.metadata.version(n) for n in ('nncase','nncase-kpu')}
    if set(versions.values())!={'2.11.0'}:raise ValueError('Compiler/runtime pin differs')
    rows=[json.loads(x) for x in (data/'manifest.jsonl').read_text().splitlines()];audit(rows,data)
    windows=build_sequences(rows,8)
    calibration=[s for s in windows if s['split']=='train'][::9][:20]
    evaluation=[s for s in windows if s['split']=='test']
    def tensor(s):
        x=[np.asarray(Image.open(data/p).convert('RGB')).transpose(2,0,1) for p in s['frames'][-args.window:]]
        return np.concatenate(x,axis=0)[None].astype(np.float32)/255
    shape=[1,3*args.window,90,160]
    opts=nncase.CompileOptions();opts.target='k230';opts.preprocess=False
    opts.input_shape=shape;opts.input_type='float32';opts.shape_bucket_enable=False
    opts.dump_ir=False;opts.dump_asm=False;opts.dump_dir=str(out/'dump')
    os.chdir(out);compiler=nncase.Compiler(opts)
    compiler.import_onnx(model.read_bytes(),nncase.ImportOptions())
    ptq=nncase.PTQTensorOptions();ptq.samples_count=len(calibration)
    ptq.calibrate_method='NoClip';ptq.quant_type='uint8';ptq.w_quant_type='uint8'
    ptq.set_tensor_data([[tensor(s)] for s in calibration]);compiler.use_ptq(ptq);compiler.compile()
    binary=compiler.gencode_tobytes();(out/'camera_xy.kmodel').write_bytes(binary)
    runtime=nncase.Simulator();runtime.load_model(binary)
    session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
    reports=[]
    for s in evaluation:
        x=tensor(s);expected=session.run(None,{'rgb_window':x})
        runtime.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));started=time.perf_counter();runtime.run()
        elapsed=time.perf_counter()-started
        actual=[runtime.get_output_tensor(i).to_numpy() for i in range(runtime.outputs_size)]
        if len(actual)!=2 or any(a.shape!=b.shape or not np.isfinite(a).all() for a,b in zip(actual,expected)):
            raise ValueError('Invalid Simulator output')
        gt=np.array([s['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')])
        def decode(v):
            idx=int(v[0].argmax());return np.array([idx%24,idx//24])*100+v[1].reshape(2)*100-1200
        reports.append({'sample_id':s['sample_id'],'trajectory_id':s['trajectory_id'],
                        'onnx_error_mm':float(np.linalg.norm(decode(expected)-gt)),
                        'kmodel_error_mm':float(np.linalg.norm(decode(actual)-gt)),
                        'host_simulator_seconds':elapsed,'grid_argmax_same':int(actual[0].argmax())==int(expected[0].argmax()),
                        'mae':[float(abs(a-b).mean()) for a,b in zip(actual,expected)]})
    report={'status':'COMPILE_AND_SIMULATOR_PASS','scope':'TRAINED_PILOT_NOT_COMPETITION_ACCEPTANCE',
            'versions':versions,'onnx_sha256':digest(model),'kmodel_sha256':digest(out/'camera_xy.kmodel'),
            'kmodel_bytes':len(binary),'dataset_manifest_sha256':digest(data/'manifest.jsonl'),
            'input':{'shape':shape,'type':'float32','layout':'packed RGB history channels NCHW',
                     'normalization':'external RGB /255 once; compiler preprocess=False'},
            'calibration_ids':[s['sample_id'] for s in calibration],'calibration_split':'train ONLY',
            'evaluation_count':len(reports),'pairs':reports,'native_k230':'NOT RUN'}
    (out/'KMODEL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','calibration_ids')}))


if __name__=='__main__':main()
